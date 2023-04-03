from celery import Celery
import time
import os

# get the hostname for the RabbitMQ service
# - or use localhost if not set (running without Docker Compose)
rabbitmq_host = os.getenv( 'RABBITMQ_HOST', 'localhost' )

# define the connection to the RabbitMQ host
app = Celery(
    'tasks',
    broker = f'pyamqp://guest@{rabbitmq_host}//',
)

# definition of the Celery application's task queue
@app.task
def run_request_task(
    data,
):
    print("Running from task definition in server? is this definition even necessary other than to register the name of the task?")
    assert False, "This code should never actually get run, unless I'm mistaken."

# add a new task to the queue
def handle_new_run_request(
    run_request,
):
    run_request_task.delay( run_request )


