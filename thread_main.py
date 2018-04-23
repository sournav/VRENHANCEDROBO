import os
import sys
import threading

def thread_process_video():
    os.system("py thread_cap.py")
def thread_detect():
    os.system("py speed_limit_search.py")
def thread_cv_op():
    os.system("py thread_cv.py")

if __name__ == "__main__":
    threads = []
    t1 = threading.Thread(target=thread_detect)
    t2 = threading.Thread(target=thread_cv_op)
    #t3 = threading.Thread(target=thread_process_video)
    
    threads.append(t1)
    threads.append(t2)
    #threads.append(t3)
    
    t1.start()
    t2.start()
    #t3.start()
    
