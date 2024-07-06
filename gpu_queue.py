from enum import Enum
from typing import List
from dataclasses import dataclass, field
from itertools import count
from strictfire import StrictFire
from datetime import timedelta
from tabulate import tabulate
from logging.handlers import RotatingFileHandler
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import re
import threading
import textwrap
import os
import socket
import json
import gpustat
import subprocess
import heapq
import time
import threading
import datetime
import logging
import psutil


# Add this near the top of your script, after the imports
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    TERMINATED = "terminated"
    DELETED = "deleted"


class TqdmFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_percentage = -1

    def filter(self, record):
        if (
            "it/s" in record.msg or "s/it" in record.msg
        ):  # This is likely a tqdm output
            match = re.search(r"(\d+)%", record.msg)
            if match:
                current_percentage = int(match.group(1))
                if current_percentage != self.last_percentage:
                    self.last_percentage = current_percentage
                    return True
                return False
        return True


@dataclass()
class Task:
    command: str
    n_gpus: int
    priority: int = 10
    output_file: str = None
    output_file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    output_file_backup_count: int = 2
    task_id: str = field(init=False)
    wait_task_ids: List[str] = field(default_factory=list)
    status: TaskStatus = field(default=TaskStatus.WAITING)
    submission_time: datetime.datetime = field(
        default_factory=datetime.datetime.now
    )
    start_time: datetime.datetime = None
    finish_time: datetime.datetime = None
    assigned_gpus: List[int] = field(default_factory=list)
    _id_counter = count(1)

    def __post_init__(self):
        task_id = next(self._id_counter)
        if self.output_file is None:
            while True:
                output_file = os.path.join("logs", f"{task_id}_out.txt")
                if os.path.exists(output_file):
                    task_id = next(self._id_counter)
                else:
                    break
            self.output_file = output_file
        self.task_id = task_id

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return (self.status, self.priority, self.task_id) == (
            other.status,
            other.priority,
            other.task_id,
        )

    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        # Waiting tasks have higher priority
        # For tasks with the same status, higher priority (larger number) comes first
        # If status and priorities are equal, earlier task_id comes first
        return (self.status != "waiting" and other.status == "waiting") or (
            (-self.priority, self.task_id) < (-other.priority, other.task_id)
        )


class GPUTaskQueue:
    """A priority queue for managing GPU tasks."""

    def __init__(
        self,
        name: str = "kf05",
        max_n_gpus: int = None,
        email_config_fn: str = "email_config.json",
    ):
        self.queue = []
        self.name = name
        self.assigned_gpus = set()
        self.running_processes = []
        self.lock = threading.Lock()
        if max_n_gpus is None:
            max_n_gpus = len(gpustat.new_query())
        self.max_n_gpus = max_n_gpus
        email_config = {}
        if os.path.exists(email_config_fn):
            with open(email_config_fn, "r") as f:
                email_config = json.load(f)
            for k in ["fromaddr", "toaddr", "password"]:
                if k not in email_config:
                    raise ValueError(
                        f"`{k}` should be provided in email config."
                    )
            logger.info("Email config has been loaded.")
        self.email_config = email_config
        logger.info(f"{self.name} is ready!")

    def sendEmail(self, task: Task):
        if self.email_config:
            fromaddr = self.email_config["fromaddr"]
            toaddr = self.email_config["toaddr"]
            msg = MIMEMultipart()
            msg["From"] = fromaddr
            msg["To"] = toaddr
            msg["Subject"] = f"A task has finished in {self.name}."

            password = self.email_config["password"]

            duration = _format_td(task.finish_time - task.start_time)
            body = ""
            body += f"Task ID: {task.task_id}\n"
            body += f"Command: {task.command}\n"
            body += f"GPUs: {task.n_gpus}\n"
            body += f"Assigned GPU IDs: {', '.join(map(str, task.assigned_gpus)) if task.assigned_gpus else 'N/A'}\n"
            body += f"Output File: {task.output_file}\n"
            body += f"Priority: {task.priority}\n"
            body += f"Wait Task IDs: {', '.join(task.wait_task_ids) if task.wait_task_ids else 'N/A'}\n"
            body += f"Status: {task.status.value}\n"
            body += f"Submitted: {task.submission_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            body += f"Started: {task.start_time.strftime('%Y-%m-%d %H:%M:%S') if task.start_time else 'N/A'}\n"
            body += f"Finished: {task.finish_time.strftime('%Y-%m-%d %H:%M:%S') if task.finish_time else 'N/A'}\n"
            body += f"Duration: {duration}\n"
            msg.attach(MIMEText(body, "plain"))
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(fromaddr, password)
            text = msg.as_string()
            server.sendmail(fromaddr, toaddr, text)
            server.quit()

    def get_available_gpus(self) -> List[int]:
        """Retrieves a list of available GPU IDs.

        Returns:
            A list of integers representing available GPU IDs.
        """
        available_gpus = []
        gpu_stats_list = gpustat.new_query()

        for gpu_id, gpu_stat in enumerate(gpu_stats_list):
            used = gpu_stat["memory.used"]
            total = gpu_stat["memory.total"]
            if used / total <= 0.05 and gpu_id not in self.assigned_gpus:
                available_gpus.append(gpu_id)
        max_n_gpus = max(0, self.max_n_gpus - len(self.assigned_gpus))
        return available_gpus[:max_n_gpus]

    def set_max_gpus(self, max_n_gpus):
        with self.lock:
            self.max_n_gpus = max_n_gpus
        logger.info(f"Max GPUs set to: {max_n_gpus}")

    def add_task(
        self,
        command: str,
        n_gpus: int,
        output_file: str = None,
        output_file_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        output_file_backup_count: int = 2,
        priority: int = 10,
        wait_task_ids: List[int] = [],
    ):
        """Adds a new task to the queue.

        Args:
            command: A string containing the command to be executed.
            n_gpus: An integer representing the number of GPUs required for the task.
            output_file: A string path where the task output will be written.
            priority: An integer representing the task's priority (default is 10).
        """
        task = Task(
            command=command,
            n_gpus=n_gpus,
            output_file=output_file,
            output_file_max_bytes=output_file_max_bytes,
            output_file_backup_count=output_file_backup_count,
            priority=priority,
            wait_task_ids=wait_task_ids,
        )
        with self.lock:
            heapq.heappush(self.queue, task)
        logger.info(
            f"Task added: ID={task.task_id}, Command='{command}', GPUs={n_gpus}, Priority={priority}, Wait Task IDs={wait_task_ids}"
        )
        return task

    def remove_deadlocks(self):
        waiting_tasks = [
            task
            for task in self.queue
            if task.status == TaskStatus.WAITING and task.wait_task_ids
        ]
        removed_tasks_id = [
            task.task_id
            for task in self.queue
            if task.status in [TaskStatus.TERMINATED, TaskStatus.DELETED]
        ]
        to_remove_tasks = []
        for task in waiting_tasks:
            for wait_task_id in task.wait_task_ids:
                if wait_task_id in removed_tasks_id:
                    to_remove_tasks.append((task, wait_task_id))
                    break

        for task, wait_task_id in to_remove_tasks:
            self.delete_task(task.task_id)
            logger.info(
                f"Task {task.task_id} is waiting for a terminated task {wait_task_id}. Thus, this task is deleted."
            )

    def run_tasks(self):
        available_gpus = self.get_available_gpus()
        n_available_gpus = len(available_gpus)

        with self.lock:
            if not self.queue:
                return

            waiting_tasks = [
                task
                for task in self.queue
                if task.status == TaskStatus.WAITING
            ]
            if not waiting_tasks:
                return

            highest_priority = max(task.priority for task in waiting_tasks)
            highest_priority_tasks = [
                task
                for task in waiting_tasks
                if task.priority == highest_priority
            ]

            task_to_run = None
            finished_tasks_id = [
                task.task_id
                for task in self.queue
                if task.status in [TaskStatus.FINISHED]
            ]
            for task in highest_priority_tasks:
                if task.n_gpus <= n_available_gpus and (
                    len(task.wait_task_ids) == 0
                    or all(t in finished_tasks_id for t in task.wait_task_ids)
                ):
                    task_to_run = task
                    break

            if task_to_run:
                selected_gpus = available_gpus[: task.n_gpus]
                self._run_task(task_to_run, selected_gpus)

    def _run_task(self, task: Task, selected_gpus: List[int]):
        """Runs a single task with output file size restriction."""
        gpu_ids_str = ",".join(map(str, selected_gpus))
        cuda_command = f"CUDA_VISIBLE_DEVICES={gpu_ids_str} {task.command}"

        # Set up output file and rotating file handler
        task_logger = self._setup_output_file(task)

        # Start the process
        process = subprocess.Popen(
            cuda_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            universal_newlines=True,
        )

        # Update task and queue information
        self._update_task_info(task, process, selected_gpus, cuda_command)

        # Start output reader thread
        self._start_output_reader(process, task_logger)

        logger.info(
            f"Started task: ID={task.task_id}, Command='{cuda_command}', "
            f"Assigned GPU IDs={selected_gpus}, Output file={task.output_file}."
        )

    def _setup_output_file(self, task: Task):
        """Set up the output file and return a task-specific logger."""
        if task.output_file:
            if directory := os.path.dirname(task.output_file):
                os.makedirs(directory, exist_ok=True)

            # Remove the existing file if it exists
            if os.path.exists(task.output_file):
                os.remove(task.output_file)

            rotating_handler = RotatingFileHandler(
                task.output_file,
                maxBytes=task.output_file_max_bytes,
                backupCount=task.output_file_backup_count,
            )
            tqdm_filter = TqdmFilter()
            rotating_handler.addFilter(tqdm_filter)

            # Create a custom logger for the task output
            task_logger = logging.getLogger(f"task_{task.task_id}")
            task_logger.setLevel(logging.INFO)
            task_logger.addHandler(rotating_handler)
            task_logger.propagate = (
                False  # Prevent propagation to parent loggers
            )
            return task_logger

    def _update_task_info(
        self, task: Task, process, selected_gpus: List[int], cuda_command: str
    ):
        """Update task and queue information after starting a process."""
        self.running_processes.append((process, task))
        self.assigned_gpus.update(selected_gpus)
        task.command = cuda_command
        task.status = TaskStatus.RUNNING
        task.assigned_gpus = selected_gpus
        task.start_time = datetime.datetime.now()

    def _start_output_reader(self, process, task_logger):
        """Start a thread to handle the process output."""

        def output_reader():
            for line in process.stdout:
                if task_logger:
                    task_logger.info(line.strip())

        threading.Thread(target=output_reader, daemon=True).start()

    def monitor_and_clean(self):
        with self.lock:
            completed_processes = []
        for process, task in self.running_processes:
            if process.poll() is not None:  # Task has completed
                task.finish_time = datetime.datetime.now()
                duration = _format_td(task.finish_time - task.start_time)
                logger.info(
                    f"Task completed: ID={task.task_id}, Command='{task.command}', Output file={task.output_file}, Duration={duration}."
                )
                task.status = TaskStatus.FINISHED
                self.sendEmail(task)
                self.assigned_gpus.difference_update(set(task.assigned_gpus))
                completed_processes.append((process, task))
        # Remove completed processes from the list
        for completed in completed_processes:
            self.running_processes.remove(completed)

    def run(self):
        """Main loop to continuously process the task queue."""
        while True:
            self.remove_deadlocks()
            self.run_tasks()
            self.monitor_and_clean()
            time.sleep(2)  # Wait for 2 seconds before next iteration

    def list_tasks(
        self,
        statuses: List[TaskStatus] = [TaskStatus.WAITING, TaskStatus.RUNNING],
    ):

        outputs = []
        for task in self.queue:
            if task.status in statuses:
                finish_time = (
                    task.finish_time.strftime("%Y-%m-%d %H:%M:%S")
                    if task.finish_time
                    else "N/A"
                )
                start_time = (
                    task.start_time.strftime("%Y-%m-%d %H:%M:%S")
                    if task.start_time
                    else "N/A"
                )
                assigned_gpus = (
                    ", ".join(map(str, task.assigned_gpus))
                    if task.assigned_gpus
                    else "N/A"
                )
                wait_task_ids = (
                    ", ".join(task.wait_task_ids)
                    if task.wait_task_ids
                    else "N/A"
                )
                # outputs.append(
                #     f"ID: {task.task_id}, Command: {task.command}, GPUs: {task.n_gpus}, "
                #     f"Assigned GPU IDs: {assigned_gpus}, Output File: {task.output_file}, Priority: {task.priority}, "
                #     f"Wait Task ID: {task.wait_task_id or 'N/A'}, "
                #     f"Status: {task.status.value}, Submitted: {task.submission_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                #     f"Finished: {finish_time}"
                # )
                outputs.append(
                    {
                        "ID": task.task_id,
                        "Command": task.command,
                        "GPUs": task.n_gpus,
                        "Assigned GPU IDs": assigned_gpus,
                        "Output File": task.output_file,
                        "Priority": task.priority,
                        "Wait Task IDs": wait_task_ids,
                        "Status": task.status.value,
                        "Submitted": task.submission_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Started": start_time,
                        "Finished": finish_time,
                    }
                )
        return outputs
        # return tabulate(outputs, headers, tablefmt="simple_grid")
        # return outputs

    def delete_task(self, task_id):
        for task in self.queue:
            if task.task_id == task_id:
                if task.status == TaskStatus.WAITING:
                    task.status = TaskStatus.DELETED
                    return True
                elif task.status == TaskStatus.RUNNING:
                    logger.info(
                        f"Task {task_id} is running. Use `terminate_task` to terminate it instead."
                    )
                else:
                    logger.info(
                        f"Task {task_id} is {task.status.value} and not deleted."
                    )
        return False

    def kill_task(self, task_id):
        with self.lock:
            for process, task in self.running_processes:
                if task.task_id == task_id:
                    try:
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            child.kill()
                        parent.kill()

                        # Wait for the process to end
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                f"Process for task {task_id} did not terminate within the timeout period."
                            )

                        self.assigned_gpus.difference_update(
                            set(task.assigned_gpus)
                        )
                        self.running_processes.remove((process, task))
                        task.status = TaskStatus.TERMINATED
                        task.finish_time = datetime.datetime.now()
                        logger.info(
                            f"Task killed: ID={task.task_id}, Command='{task.command}'"
                        )
                        return True
                    except psutil.NoSuchProcess:
                        logger.warning(
                            f"Process for task {task_id} no longer exists."
                        )
                        return False
        return False


def handle_client(client_socket, gpu_queue):
    while True:
        try:
            data = client_socket.recv(1024).decode("utf-8")
            if not data:
                break

            request = json.loads(data)
            action = request.get("action")

            if action == "add":
                task = gpu_queue.add_task(
                    command=request["command"],
                    n_gpus=request["n_gpus"],
                    output_file=request["output_file"],
                    output_file_max_bytes=request["output_file_max_bytes"],
                    output_file_backup_count=request[
                        "output_file_backup_count"
                    ],
                    priority=request["priority"],
                    wait_task_ids=request["wait_task_ids"],
                )
                response = {
                    "status": "success",
                    "message": f"Task added successfully with ID: {task.task_id}",
                }
            elif action == "list":
                statuses = [TaskStatus(s) for s in request["statuses"]]
                tasks = gpu_queue.list_tasks(statuses=statuses)
                response = {
                    "status": "success",
                    "tasks": tasks,
                    "message": "List tasks to client.",
                }
            elif action == "delete":
                success = gpu_queue.delete_task(request["task_id"])
                if success:
                    response = {
                        "status": "success",
                        "message": f"Task {request['task_id']} deleted successfully",
                    }
                else:
                    response = {
                        "status": "error",
                        "message": f"Task {request['task_id']} not found",
                    }
            elif action == "kill":
                success = gpu_queue.kill_task(request["task_id"])
                if success:
                    response = {
                        "status": "success",
                        "message": f"Task {request['task_id']} killed successfully",
                    }
                else:
                    response = {
                        "status": "error",
                        "message": f"Task {request['task_id']} not found or not running",
                    }
            elif action == "set_max_gpus":
                max_gpus = request.get("max_n_gpus")
                gpu_queue.set_max_gpus(max_gpus)
                response = {
                    "status": "success",
                    "message": f"Max number of GPUs to used set to {max_gpus}",
                }
            else:
                response = {"status": "error", "message": "Invalid action"}

            print(response["message"])
            client_socket.send(json.dumps(response).encode("utf-8"))
        except Exception as e:
            print(f"Error handling client: {e}")
            break

    client_socket.close()


def _run_server(host, port, gpu_queue):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        # print(f"Accepted connection from {addr}")
        client_thread = threading.Thread(
            target=handle_client, args=(client_socket, gpu_queue)
        )
        client_thread.start()


def _receive_json(sock):
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            # If it's not valid JSON yet, we need more data
            continue


def _send_request(host, port, request):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    client_socket.send(json.dumps(request).encode("utf-8"))
    response = _receive_json(client_socket)
    # response = json.loads(client_socket.recv(1024).decode("utf-8"))

    client_socket.close()
    return response


def _format_td(td: timedelta) -> str:
    days = td.days
    total_seconds = td.seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"


def add_task(
    command: str,
    n_gpus: int,
    output_file: str = None,
    output_file_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    output_file_backup_count: int = 2,
    priority: int = 10,
    wait_task_ids: List[int] = [],
    host="localhost",
    port=12345,
):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    request = {
        "action": "add",
        "command": command,
        "n_gpus": n_gpus,
        "output_file": output_file,
        "output_file_max_bytes": output_file_max_bytes,
        "output_file_backup_count": output_file_backup_count,
        "priority": priority,
        "wait_task_ids": wait_task_ids,
    }

    response = _send_request(host, port, request)
    print(f"Server response: {response['message']}")


def list_tasks(
    task_types: str = "running,waiting", host="localhost", port=12345
):
    if task_types == "all":
        statuses = [status.value for status in TaskStatus]
    else:
        statuses = [
            TaskStatus(task_type).value for task_type in task_types.split(",")
        ]
    request = {"action": "list", "statuses": statuses}
    response = _send_request(host, port, request)
    if response["status"] == "success":
        for t in response["tasks"]:
            # command
            t["Command"] = textwrap.fill(t["Command"], width=40)
            t["Output File"] = textwrap.fill(t["Output File"], width=40)
            for k in ["Submitted", "Started", "Finished"]:
                t[k] = "\n".join(t[k].split())

        output_str = tabulate(
            response["tasks"], tablefmt="simple_grid", headers="keys"
        )
        print(output_str)
    else:
        print(f"Error: {response['message']}")


def delete_task(task_id, host="localhost", port=12345):
    request = {"action": "delete", "task_id": task_id}
    response = _send_request(host, port, request)
    print(f"Server response: {response['message']}")


def kill_task(task_id, host="localhost", port=12345):
    request = {"action": "kill", "task_id": task_id}
    response = _send_request(host, port, request)
    print(f"Server response: {response['message']}")


def set_max_n_gpus(max_n_gpus: int, host="localhost", port=12345):
    request = {"action": "set_max_gpus", "max_n_gpus": max_n_gpus}
    response = _send_request(host, port, request)
    print(f"Server response: {response['message']}")


def run_server(name: str = "kf05", email_config_fn: str = None, port:int=12345):
    if email_config_fn is None:
        parent_folder = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(parent_folder, "email_config.json")
        if os.path.exists(fn):
            email_config_fn = fn
    gpu_queue = GPUTaskQueue(name=name, email_config_fn=email_config_fn)
    queue_thread = threading.Thread(target=gpu_queue.run, daemon=True)
    queue_thread.start()
    # Run the server
    _run_server("localhost", port, gpu_queue)


if __name__ == "__main__":
    StrictFire(
        {
            "run_server": run_server,
            "set_max_n_gpus": set_max_n_gpus,
            "add": add_task,
            "kill": kill_task,
            "delete": delete_task,
            "list": list_tasks,
        }
    )
