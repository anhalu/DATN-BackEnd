

# Build docker 


1. Export conda environment
```shell
conda env export --no-builds   | grep -v "^prefix: " > environment.yml   
```
2. Build docker
```shell
docker build --platform linux/amd64 -t registry-dev.tcgroup.vn/ocr_core --progress plain .  
```
3. Thư viện yêu cầu:
```sh
conda install --channel conda-forge pyvips
```
4. 