import glob
import json
import logging
import os.path
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Optional

import aiofiles
import cv2
import numpy as np
import requests
import sentry_sdk
import uvicorn
from decouple import config
from fastapi import Depends, FastAPI, HTTPException, status, Body, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse, JSONResponse
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND
from typing_extensions import Annotated

import utils
from application import crud
from application import models
from application import schemas
from application.database import SessionLocal, engine
from application.schemas import FormParseResponse
from content_parser.classify import classify_using_yolo
from content_parser.rule_v6 import parse_docs_using_yolo
from content_parser.rule_v7 import ContentParserV7
from layout_parser.document_builder import export_pdf_searchable, export_docs
from layout_parser.elements import Document
from layout_parser.pdf2docs import Pdf2Docs
from reader import Reader

sentry_sdk.init(
    dsn="https://9a7e0b9b7dd3b92612a6e8cec19fd9a7@o4506580158316544.ingest.sentry.io/4506580482064384",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

models.Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = config("SECRET_KEY", default="6d19701218add6833f638c47663f54e86402726cb88a87d4f6242c675321b30d")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=129600, cast=int)
MASTER_KEY = config("MASTER_KEY", default="thiennt")
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_BASE_DIR = config("IMAGE_BASE_DIR", default=f"{BASE_DIR}/data/image/requests")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(db, username: str, password: str):
    user = crud.get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)],
                           db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = crud.get_user(db, token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
        current_user: Annotated[schemas.User, Depends(get_current_user)]
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def redirect_to_docs():
    return RedirectResponse(url='/docs')


@app.post("/create-user", response_model=schemas.User, status_code=HTTP_201_CREATED, tags=['user'])
async def create_user_api(new_user: schemas.UserCreate = Body(..., embed=True), db: Session = Depends(get_db)):
    if new_user.master_key != MASTER_KEY:
        raise HTTPException(status_code=401, detail="Permission deny")
    db_user = crud.get_user(db, new_user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(new_user.password)
    return crud.create_user(db, new_user, hashed_password)


@app.post("/token", response_model=schemas.Token, tags=['user'])
async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=schemas.User, tags=['user'])
async def read_users_me(
        current_user: Annotated[schemas.User, Depends(get_current_active_user)],
):
    return current_user


# @app.get("/api/v1/file/list", response_model=List[schemas.File], tags=['file'])
# async def read_own_files(
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
#         skip: int = 0, limit: int = 100,
# ):
#     files = crud.get_files(db, current_user.id, skip, limit)
#     return files
#
#
# @app.get("/api/v1/file/{file_id}", tags=['file'])
# async def read_own_file(
#         file_id: str,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     file = crud.get_file(db, current_user.id, file_id)
#
#     return FileResponse(file.file_path, filename=file.filename)
#
#
# @app.get("/api/v1/file/{file_id}/images", tags=['file'])
# async def read_image_of_file(
#         file_id: str,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     file = crud.get_file(db, current_user.id, file_id)
#     file_id = file.file_id
#     folder = os.path.dirname(file.file_path)
#     files = list(sorted(glob.glob(f'{folder}/{file_id}_*.jpg')))
#     files = [os.path.basename(f) for f in files]
#     return {
#         "images": files
#     }
#
#
# @app.get("/api/v1/file/{file_id}/image/{image_name}", tags=['file'])
# async def read_image_of_file(
#         file_id: str,
#         image_name: str,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     file = crud.get_file(db, current_user.id, file_id)
#     file_id = file.file_id
#     folder = os.path.dirname(file.file_path)
#     file = f'{folder}/{image_name}'
#     return FileResponse(path=file, filename=image_name)


@app.post("/api/v1/file/upload", tags=['file'])
async def upload_file(current_user: Annotated[schemas.User, Depends(get_current_active_user)],
                      db: Session = Depends(get_db),
                      file: UploadFile = File(...), ):
    filename = file.filename
    _, file_ext = os.path.splitext(filename)
    file_ext = file_ext[1:]
    file_id = str(uuid.uuid4())
    file_path = f"{IMAGE_BASE_DIR}/{file_id}.{file_ext}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    # log request file to database
    file_create = schemas.FileCreate(file_id=file_id, filename=filename, file_ext=file_ext, file_path=file_path,
                                     file_size=file.size, status='uploaded', split_pages=False)

    crud.create_user_file(db, file_create, current_user.id)

    return {
        "file_id": file_id,
        "file_path": file_path,
        "file_name": filename
    }


@app.post("/api/v1/file/upload-minio", tags=['link'])
async def upload_file_minio(current_user: Annotated[schemas.User, Depends(get_current_active_user)],
                            request_body: schemas.MinioRequest,
                            db: Session = Depends(get_db)
                            ):
    file = request_body.file
    filename = request_body.filename
    filesize = request_body.filesize
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Not found file",
        )

    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Not found filename : {filename}"
        )

    _, file_ext = os.path.splitext(filename)
    file_ext = file_ext[1:]
    file_id = str(uuid.uuid4())
    file_path = f"{IMAGE_BASE_DIR}/{file_id}.{file_ext}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        url = file
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)
        file_byte = response.content
        # content = await file_byte
        await out_file.write(file_byte)  # async write
    # log request file to database
    file_create = schemas.FileCreate(file_id=file_id, filename=filename, file_ext=file_ext, file_path=file_path,
                                     file_size=filesize, status='uploaded', split_pages=False)
    crud.create_user_file(db, file_create, current_user.id)
    return {
        "file_id": file_id,
        "file_path": file_path,
        "file_name": filename
    }


# init model
reader_model: Reader = None


@app.on_event("startup")
def init_model():
    logger.info("Init model ....")
    global reader_model
    reader_model = Reader(line_rec=config('LINE_REC', default=True, cast=bool))

    log = logging.getLogger("uvicorn.access")
    console_formatter = uvicorn.logging.ColourizedFormatter(
        "{asctime} | {levelname: <8} | {name}: {message}",
        style="{", use_colors=True)
    log.handlers[0].setFormatter(console_formatter)


# @app.post("/api/v1/ocr/general_with_only_image", tags=['ocr'])
# async def ocr_with_only_image(
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
#         file: UploadFile = File(...)
# ):
#     if file is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Not found file id: {file.filename}",
#         )
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     images = [cv2.imdecode(nparr, cv2.IMREAD_COLOR)]
#     document: Document = reader_model.read_images(images)
#     res = document.export()
#     res['content'] = document.render()
#     return res


# @app.post("/api/v1/ocr/general_with_only_image_paddle", tags=['ocr'])
# async def general_with_only_image_paddle(
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
#         file: UploadFile = File(...)
# ):
#     if file is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Not found file id: {file.filename}",
#         )
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     images = [cv2.imdecode(nparr, cv2.IMREAD_COLOR)]
#     res = {'line': []}
#     det_results = reader_model.det_model(images)
#     for page, image in zip(det_results, images):
#         for cls, cls_name, prob, box, crop_img in page:
#             box = np.array(box).tolist()
#             res['line'].append(box)
#     return res


# @app.post("/api/v1/ocr/run_ocr", tags=['ocr'])
# async def run_ocr(
#         files: List[UploadFile],
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     images = []
#     for file in files:
#         contents = await file.read()
#         nparr = np.fromstring(contents, np.uint8)
#         images.append(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
#     texts, probs, _ = reader_model._recognite_text(images)
#     data = []
#     for text, prob in zip(texts, probs):
#         data.append([text, prob])
#     return data


@app.post("/api/v1/ocr/general", tags=['ocr'])
async def ocr_general(
        file_id: str,
        current_user: Annotated[schemas.User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
        export_file: bool = False,
        cached: bool = True
):
    start = time.time()
    # extract image
    file = crud.get_file(db, current_user.id, file_id)
    if file is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Not found file id: {file_id}",
        )
    filename = file.filename
    file_path = file.file_path
    file_size = file.file_size
    file_ext = file.file_ext
    status = file.status
    split_pages = file.split_pages
    logger.info(f"Get file {file_id}: {filename}")

    if cached and os.path.exists(f"{IMAGE_BASE_DIR}/{file_id}.json"):
        logger.info(f"File {file_id} has process before: loaded saved result.")
        try:
            return json.load(open(f"{IMAGE_BASE_DIR}/{file_id}.json"))
        except Exception as e:
            logger.error(f"Load json failed: {e}")

    if not split_pages and status == 'processing':
        return JSONResponse(status_code=202, content={
            "error_code": 1,
            "message": "File still processing..."
        })

    file.status = 'processing'
    crud.save_file(db, file)

    BATCH_SIZE = 16
    batch_images = []
    document: Optional[Document] = None
    layout_predict = {}
    batch_idx = 0
    start_for = time.time()

    def process_batch(document):
        logger.info(f"Process batch {batch_idx}: {len(batch_images)}")
        read_image_time = time.time()
        document_batch: Document = reader_model.read_images(batch_images, file_id=file_id)
        logger.info(f'TIME READ IMAGE PREDICT {len(batch_images)} images : {time.time() - read_image_time} ')
        if document is None:
            document = document_batch
        else:
            for idx_batch in range(len(batch_images)):
                document_batch.pages[idx_batch].page_idx = batch_idx * BATCH_SIZE + idx_batch
            document.pages.extend(document_batch.pages)
        layout_start_time = time.time()
        layout_predict_batch = reader_model.predict_layout(batch_images, document)
        logger.info(f'TIME LAYOUT PREDICT {len(batch_images)} images : {time.time() - layout_start_time}')
        for (p, r), page in zip(layout_predict_batch.items(), document_batch.pages):
            layout_predict[len(layout_predict)] = r
            page.figures.extend(r.get('figures', []))
            page.tables.extend(r.get('tables', []))
            page.signature_boxes.extend(r.get('signature_boxes', []))
            page.title_box = r.get('titles', None)
            page.page_number = r.get('page_numbers', None)
        batch_images.clear()
        return document

    for i, image in enumerate(utils.get_images_local(file_path, file_id, file_ext, IMAGE_BASE_DIR)):
        if i < 1 and not split_pages:
            file.split_pages = True
            crud.save_file(db, file)
        batch_images.append(image)
        if len(batch_images) >= BATCH_SIZE:
            document = process_batch(document)
            batch_idx += 1

    logger.info(f"Time for : {time.time() - start_for}")
    if len(batch_images) > 0:
        document = process_batch(document)

    if document is not None:
        logger.info(f"File {file_id} has {len(document.pages)} pages")
        file.status = 'processed'
        file.split_pages = True
        crud.save_file(db, file)
    else:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Processing failed: {file_id}",
        )

    logger.info(f"File {file_id} docs: {document}")
    document.remove_text_in_figure()
    res = document.export()
    res['content'] = document.render()
    res['response_layout'] = layout_predict

    json.dump(res, open(f"{IMAGE_BASE_DIR}/{file_id}.json", 'w', encoding="utf8"), ensure_ascii=False)
    if export_file:
        return document
    logger.info(f"TIME RUN OCR_GENERAL : {time.time() - start}")

    return res


# @app.post("/api/v1/ocr/general/export/docx", tags=['ocr'])
# async def ocr_parse_docx(
#         file_id: str,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     file = crud.get_file(db, current_user.id, file_id)
#     filename = file.filename
#     filename_output = f'{os.path.splitext(filename)[0]}.docx'
#     document = await ocr_general(file_id, current_user, db, export_file=True)
#     if isinstance(document, dict):
#         document = Document.from_dict(document)
#         image_paths = sorted(glob.glob(f'{IMAGE_BASE_DIR}/{file_id}_*.jpg'))
#         for i, page in enumerate(document.pages):
#             img = cv2.imread(image_paths[i])
#             page.image = img
#     output_path = tempfile.mktemp()
#     export_docs(document, output_path)
#     return FileResponse(output_path, filename=filename_output)
#
#
# @app.post("/api/v1/ocr/general/export/pdf", tags=['ocr'])
# async def ocr_parse_pdf(
#         file_id: str,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     file = crud.get_file(db, current_user.id, file_id)
#     filename = file.filename
#     document = await ocr_general(file_id, current_user, db, export_file=True)
#     if isinstance(document, dict):
#         document = Document.from_dict(document)
#         image_paths = sorted(glob.glob(f'{IMAGE_BASE_DIR}/{file_id}_*.jpg'))
#         for i, page in enumerate(document.pages):
#             img = cv2.imread(image_paths[i])
#             page.image = img
#     output_path = tempfile.mktemp()
#     export_pdf_searchable(document, output_path)
#     return FileResponse(output_path, filename=filename)


@app.post("/api/v1/ocr/parse-with-rule-using-layout", tags=['ocr'], response_model=schemas.FormParseResponse)
async def ocr_form_with_rule_using_layout(
        request_body: schemas.FormParseRequest,
        current_user: Annotated[schemas.User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
):
    # TODO:
    # extract image

    file_id = request_body.request_id

    file = crud.get_file(db, current_user.id, file_id)
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Not found file id: {file_id}",
        )
    filename = file.filename
    file_path = file.file_path
    file_size = file.file_size
    file_ext = file.file_ext
    # logger.debug(f"ocr_from input: {request_body}")

    if os.path.exists(f"{IMAGE_BASE_DIR}/{file_id}.json"):
        results_reponse = parse_docs_using_yolo(file=f"{IMAGE_BASE_DIR}/{file_id}.json",
                                                rule=jsonable_encoder(request_body))

        return FormParseResponse(**results_reponse)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Not found file id: {file_id}",
    )


@app.post("/api/v1/ocr/parse-with-group-rule", tags=['ocr'], response_model=List[schemas.FormValue])
async def ocr_form_with_group_rule(
        request_body: schemas.GroupFormParseRequest,
        # request_body: schemas.FormParseRequest_classify,
        current_user: Annotated[schemas.User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
):
    file_id = request_body.request_id

    file = crud.get_file(db, current_user.id, file_id)
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Not found file id: {file_id}",
        )
    filename = file.filename
    file_path = file.file_path
    file_size = file.file_size
    file_ext = file.file_ext
    # logger.debug(f"ocr_from input: {request_body}")

    ocr_result_path = f"{IMAGE_BASE_DIR}/{file_id}.json"
    if os.path.exists(ocr_result_path):
        document = Document.from_dict(json.load(open(ocr_result_path, 'r', encoding='utf8')))
        logger.debug(f"Ocr from with group rule for document: {document.info}")
        # new_request = schemas.convert_old_input_body(request_body)
        new_request = request_body
        parser = ContentParserV7()
        return parser(document=document, group_forms=new_request.group_forms, from_page=new_request.from_page_idx,
                      to_page=new_request.to_page_idx)

        # return FormParseResponse(**results_reponse)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Not complete ocr general: {file_id}",
    )


# @app.post("/api/v1/ocr/classify", tags=['ocr'], response_model=schemas.FormParseResponse_classify)
# async def ocr_form_with_rule_classify(
#         request_body: schemas.FormParseRequest_classify,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
# ):
#     # TODO:
#     # extract image
#     s = time.time()
#     file_id = request_body.request_id
#
#     file = crud.get_file(db, current_user.id, file_id)
#     if file is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Not found file id: {file_id}",
#         )
#     filename = file.filename
#     file_path = file.file_path
#     file_size = file.file_size
#     file_ext = file.file_ext
#     # logger.debug(f"ocr_from input: {request_body}")
#
#     if os.path.exists(f"{IMAGE_BASE_DIR}/{file_id}.json"):
#         results_reponse = classify_using_yolo(file=f"{IMAGE_BASE_DIR}/{file_id}.json",
#                                               rule=jsonable_encoder(request_body))
#
#         logger.info(f"------------------------------Time Classify : {time.time() - s}")
#         return schemas.FormParseResponse_classify(**results_reponse)
#
#     raise HTTPException(
#         status_code=status.HTTP_404_NOT_FOUND,
#         detail=f"Not found file id: {file_id}",
#     )
#
#
# @app.post("/api/v1/ocr/export-file-txt", tags=['demo'])
# async def export_file_txt(
#         request_body: schemas.ExportTxtRequest,
#         current_user: Annotated[schemas.User, Depends(get_current_active_user)],
#         db: Session = Depends(get_db),
#         cached: bool = True
# ):
#     file_id = request_body.request_id
#     file = crud.get_file(db, current_user.id, file_id)
#     if file is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Not found file id: {file_id}",
#         )
#
#     if os.path.exists(f"{IMAGE_BASE_DIR}/{file_id}_export.txt"):
#         if cached:
#             results_str = open(f"{IMAGE_BASE_DIR}/{file_id}_export.txt").read()
#             results_dict = {
#                 'text': results_str
#             }
#             return results_dict
#     if os.path.exists(f"{IMAGE_BASE_DIR}/{file_id}.json"):
#         results_str = Pdf2Docs(request_id=file_id).run()
#         results_dict = {
#             'text': results_str
#         }
#         with open(f"{IMAGE_BASE_DIR}/{file_id}_export.txt", 'w') as f:
#             f.write(results_str)
#         return results_dict
#

if __name__ == '__main__':
    uvicorn.run('api:app', reload=True)
