ps -ef | grep python | grep run.py | awk '{print "kill -9 ", $2}' > kill_process_python_run.sh
chmod 777 kill_process_python_run.sh
./kill_process_python_run.sh

rm -rf kill_process_python_run.sh

nohup python run.py >> logs/ocr_core.log &
sleep 2

tail -f logs/ocr_core.log