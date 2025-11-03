import datetime
import uuid

class Task:
    def __init__(self, user, dir, task):
        self.task_id = uuid.uuid4()
        self.user = user
        self.dir = dir
        self.task = task
        # when it is queued is considered as 
        self.user_submit_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.start_service_time = None
        self.finish_time = None
        self.status = "queued"
        self.finish_status = None
        self.recovered = False

    def _to_string(self):
        return f"task id: {self.task_id} \nsubmitted by user: {self.user}, \ndirectory: {self.dir} \ntask: {self.task} \nrecovered?: {self.recovered}"

    def set_id(self, id):
        self.task_id = id

    def set_status(self, st):
        self.status = st

    def set_if_recovered(self):
        self.recovered = True

    def set_service_time(self, sst):
        self.start_service_time = sst
    
    def set_finish_time(self, ft):
        self.finish_time = ft

    def set_finish_status(self, fs):
        self.finish_status = fs

# The queue for keeping submitted tasks
class Tasks():
    def __init__(self):
        self.queue = []
 
    def enqueue(self, value):
        # Inserting to the end of the queue
        self.queue.append(value)
 
    def dequeue(self):
         # Remove the furthest element from the top,
         # since the Queue is a FIFO structure
         return self.queue.pop(0)
    
    def check(self):
        return self.queue[0]
    
    def put_it_back(self, value):
        self.queue.insert(0, value)

    # returns the number of tasks in the queue
    def length(self):
        return len(self.queue)
    
    # returns whole queue
    def whole_list(self):
        return self.queue