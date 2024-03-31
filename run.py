if __name__ == '__main__':
    import uvicorn

    uvicorn.run('application.api:app', host='0.0.0.0', reload=False, workers=1)
