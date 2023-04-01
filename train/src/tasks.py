from celery import Celery
import time

app = Celery(
    'tasks',
    broker = 'pyamqp://guest@localhost//',
)

@app.task
def run_request_task(
    data,
):
    print( "Received %s" % data )
    time.sleep( 10 )
    print( "Completed %s" % data )

def handle_new_run_request(
    run_request,
):
    run_request_task.delay( run_request )


