import pytest
import tempfile
import os
import time
import logging
import os
from io import StringIO
from unittest.mock import Mock, patch, MagicMock
from gpu_queue import GPUTaskQueue, Task, TaskStatus, TqdmFilter


@pytest.fixture
def mock_gpustat():
    with patch("gpustat.new_query") as mock:
        mock.return_value = [
            {"memory.used": 1000, "memory.total": 10000},
            {"memory.used": 8000, "memory.total": 10000},
            {"memory.used": 500, "memory.total": 10000},
        ]
        yield mock


@pytest.fixture
def gpu_queue():
    return GPUTaskQueue()


def test_get_available_gpus(gpu_queue, mock_gpustat):
    available_gpus = gpu_queue.get_available_gpus()
    assert available_gpus == [0, 2]


def test_add_task(gpu_queue):
    command = "echo test"
    priority = 5
    gpu_queue.add_task(
        command=command, n_gpus=1, output_file="output.txt", priority=priority
    )
    assert len(gpu_queue.queue) == 1
    task = gpu_queue.queue[0]
    assert isinstance(task, Task)
    assert task.priority == 5
    assert task.command == command


def test_run_tasks_with_highest_priority(gpu_queue, mock_gpustat):
    with patch.object(gpu_queue, "_run_task") as mock_run:
        gpu_queue.add_task(
            command="echo test1",
            n_gpus=1,
            output_file="output1.txt",
            priority=1,
        )
        gpu_queue.add_task(
            command="echo test2",
            n_gpus=3,
            output_file="output2.txt",
            priority=3,
        )
        gpu_queue.add_task(
            command="echo test3",
            n_gpus=1,
            output_file="output3.txt",
            priority=5,
        )

        gpu_queue.run_tasks()

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0].command == "echo test3"


def test_run_tasks_without_available_task_with_highest_priority(
    gpu_queue, mock_gpustat
):
    with patch.object(gpu_queue, "_run_task") as mock_run:
        gpu_queue.add_task(
            command="echo test1",
            n_gpus=3,
            output_file="output1.txt",
            priority=5,
        )
        gpu_queue.add_task(
            command="echo test2",
            n_gpus=1,
            output_file="output2.txt",
            priority=3,
        )
        gpu_queue.add_task(
            command="echo test3",
            n_gpus=3,
            output_file="output3.txt",
            priority=5,
        )

        gpu_queue.run_tasks()
        assert mock_run.call_count == 0


@patch("subprocess.Popen")
def test_monitor_and_clean(mock_popen, gpu_queue):
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_process.stdout.readline.side_effect = [
        "test\n",
        "",
    ]  # return a string

    task = Task(
        command="echo test", n_gpus=1, output_file="output.txt", priority=5
    )
    gpu_queue._run_task(task, [0])

    gpu_queue.monitor_and_clean()
    assert len(gpu_queue.running_processes) == 0


@pytest.fixture
def setup_test_environment(tmp_path):
    test_output_dir = tmp_path / "logs"
    test_output_dir.mkdir(exist_ok=True)
    output_file = test_output_dir / "test_output.log"
    max_size = 1000  # 1KB for testing purposes
    backup_count = 2
    return test_output_dir, output_file, max_size, backup_count


def outputs_in_each_file(lines, byte_sizes, max_bytes, backup_count):
    files = []
    current_file = []
    current_file_size = 0

    for i, (line, byte_size) in enumerate(zip(lines, byte_sizes)):
        if current_file_size + byte_size > max_bytes:
            files.append(current_file)
            current_file = [line]
            current_file_size = byte_size
        else:
            current_file.append(line)
            current_file_size += byte_size

        if len(files) > backup_count:
            files.pop(0)

    if current_file:
        files.append(current_file)

    return ["\n".join(file) + "\n" for file in files]


def run_rotating_file_handler_test(gpu_queue, setup, num_lines):
    test_output_dir, output_file, max_size, backup_count = setup

    # Create a task with a small command that generates output
    task = Task(
        command=f"python -c \"for i in range({num_lines}): print(f'Line {{i}}' * 10)\"",
        n_gpus=1,
        output_file=str(output_file),
        output_file_max_bytes=max_size,
        output_file_backup_count=backup_count,
    )

    # Run the task
    gpu_queue._run_task(task, [0])

    # Wait for the task to complete
    for process, _ in gpu_queue.running_processes:
        process.wait()

    # Force flush and wait a bit to ensure all data is written
    for handler in logging.getLogger(f"task_{task.task_id}").handlers:
        handler.flush()
    time.sleep(1)  # Give a little time for OS to finish writing

    # Check if the main log file exists and its size
    assert output_file.exists()
    assert output_file.stat().st_size <= max_size

    # expected outputs
    lines = [f"Line {i}" * 10 for i in range(num_lines)]
    byte_sizes = [len(l) + 1 for l in lines]
    expected_outputs = outputs_in_each_file(
        lines, byte_sizes, max_size, backup_count
    )
    n_expected_files = len(expected_outputs)

    # Check if backup files were created
    for i in range(1, n_expected_files):
        backup_file = output_file.with_suffix(f".log.{i}")
        assert backup_file.exists()

    # Check if the number of backup files is correct
    log_files = list(test_output_dir.glob("test_output.log*"))
    assert len(log_files) == n_expected_files  # Main log + backups

    # Read all log content
    actual_outputs = []
    for i, log_file in enumerate(
        sorted(log_files, key=lambda x: x.stat().st_mtime)
    ):
        actual_outputs.append(log_file.read_text())

    # Check if all lines are present and in order
    for actual_output, expected_output in zip(
        actual_outputs, expected_outputs
    ):
        assert actual_output == expected_output

    # Clean up
    for log_file in log_files:
        log_file.unlink()


def test_rotating_file_handler_1000_lines(gpu_queue, setup_test_environment):
    run_rotating_file_handler_test(gpu_queue, setup_test_environment, 1000)


def test_rotating_file_handler_20_lines(gpu_queue, setup_test_environment):
    run_rotating_file_handler_test(gpu_queue, setup_test_environment, 20)


###


def test_tqdm_filter_logging(gpu_queue, tmpdir):
    # Setup a StringIO to capture log output
    mock_task = Task(
        command="echo test", n_gpus=1, output_file=tmpdir / "test_output.log"
    )

    # Setup the task logger with our custom filter
    task_logger = gpu_queue._setup_output_file(mock_task)

    # Simulate tqdm output
    tqdm_outputs = [
        "10%|██        | 100/1000 [00:01<00:09, 98.78it/s]",
        "10%|██        | 101/1000 [00:01<00:09, 98.80it/s]",
        "11%|██        | 110/1000 [00:01<00:09, 98.82it/s]",
        "20%|████      | 200/1000 [00:02<00:08, 98.85it/s]",
    ]

    # Log the tqdm outputs
    for output in tqdm_outputs:
        task_logger.info(output)

    # Read the log file
    with open(mock_task.output_file, "r") as f:
        log_output = f.read().strip().split("\n")

    # Check that only percentage changes were logged
    assert len(log_output) == 3
    assert "10%" in log_output[0]
    assert "11%" in log_output[1]
    assert "20%" in log_output[2]


def test_run_tasks_with_wait_task_ids(mock_gpustat, gpu_queue):
    mock_gpustat.return_value = [
        {"memory.used": 0, "memory.total": 100} for _ in range(3)
    ]
    # Add three tasks
    task1 = gpu_queue.add_task("task1", n_gpus=1, output_file="out1.txt")
    task2 = gpu_queue.add_task("task2", n_gpus=1, output_file="out2.txt")
    task3 = gpu_queue.add_task(
        "task3",
        n_gpus=1,
        output_file="out3.txt",
        wait_task_ids=[task1.task_id, task2.task_id],
    )

    # Initial run
    for _ in range(3):
        gpu_queue.run_tasks()
    assert task1.status == TaskStatus.RUNNING
    assert task2.status == TaskStatus.RUNNING
    assert task3.status == TaskStatus.WAITING

    # Set task1 to finished
    task1.status = TaskStatus.FINISHED
    gpu_queue.run_tasks()
    assert task3.status == TaskStatus.WAITING

    # Set task2 to finished
    task2.status = TaskStatus.FINISHED
    gpu_queue.run_tasks()
    assert task3.status == TaskStatus.RUNNING


def test_remove_deadlocks(gpu_queue):
    # mock_gpustat.return_value = [
    #     {"memory.used": 0, "memory.total": 100} for _ in range(3)
    # ]
    task1 = gpu_queue.add_task("task1", n_gpus=1, output_file="out1.txt")
    task2 = gpu_queue.add_task("task2", n_gpus=1, output_file="out2.txt")
    task3 = gpu_queue.add_task(
        "task3",
        n_gpus=1,
        output_file="out3.txt",
        wait_task_ids=[task1.task_id, task2.task_id],
    )
    task4 = gpu_queue.add_task("task4", n_gpus=1, output_file="out3.txt")
    task5 = gpu_queue.add_task(
        "task5",
        n_gpus=1,
        output_file="out3.txt",
        wait_task_ids=[task4.task_id],
    )

    gpu_queue.remove_deadlocks()
    waiting_tasks = [
        task.task_id
        for task in gpu_queue.queue
        if task.status == TaskStatus.WAITING
    ]
    assert len(waiting_tasks) == 5

    gpu_queue._run_task(task2, [0])
    gpu_queue.kill_task(task2.task_id)
    assert task2.status == TaskStatus.TERMINATED
    waiting_tasks = [
        task for task in gpu_queue.queue if task.status == TaskStatus.WAITING
    ]
    assert len(waiting_tasks) == 4
    assert task2 not in waiting_tasks

    gpu_queue.remove_deadlocks()
    waiting_tasks = [
        task for task in gpu_queue.queue if task.status == TaskStatus.WAITING
    ]
    assert len(waiting_tasks) == 3
    assert task2 not in waiting_tasks
    assert task3 not in waiting_tasks

    gpu_queue.delete_task(task4.task_id)
    gpu_queue.remove_deadlocks()
    assert task4.status == TaskStatus.DELETED
    waiting_tasks = [
        task for task in gpu_queue.queue if task.status == TaskStatus.WAITING
    ]
    assert len(waiting_tasks) == 1
    assert task2 not in waiting_tasks
    assert task3 not in waiting_tasks
    assert task4 not in waiting_tasks
    assert task5 not in waiting_tasks


if __name__ == "__main__":
    pytest.main("tests/test_gpu_queue.py::test_remove_deadlocks".split(" "))
