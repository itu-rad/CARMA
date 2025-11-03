import socket
from subprocess import Popen, PIPE
import click
import os

# developed and tested
def execute_command(cmd, shell=False, vars={}):
    """
    output: string format terminal based output of a bash shell command
    Executes a bash command and return the result
    """
    cmd = cmd.split()
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True, shell=shell) as p:
        o=p.stdout.read()
    return o

@click.command()
@click.option('--task', default="task.txt", help='address of the task description')
def submit(task):
    host = socket.gethostname()  # as both code is running on same pc
    port = 5001  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    user = execute_command("whoami").strip()
    current_directory = execute_command("pwd").strip()

    # making the path absolute, so it can be used easily by the resource manager later
    task = os.path.abspath(task)

    # giving user, current directory, and task's absolute path to the resource manager
    task = user + "+" +current_directory + "+" + task

    client_socket.send(task.encode())  # send message
    
    data = client_socket.recv(1024).decode()  # receive response
    print(data)  # show in terminal
    client_socket.close()  # close the connection


if __name__ == '__main__':
    submit()