from celery import Celery
import time
import os

# get the hostname for the RabbitMQ service
# - or use localhost if not set (running without Docker Compose)
rabbitmq_host = os.getenv( 'RABBITMQ_HOST', 'localhost' )

app = Celery(
    'tasks',
    broker = f'pyamqp://guest@{rabbitmq_host}//',
)

@app.task
def run_request_task(
    data,
):
    print("Running from task definition in server? is this definition even necessary other than to register the name of the task?")

def handle_new_run_request(
    run_request,
):
    run_request_task.delay( run_request )


