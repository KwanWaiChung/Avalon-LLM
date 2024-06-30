import pytest
import tempfile
import os
import time
import io
import logging
from unittest.mock import Mock, patch, MagicMock
from gpu_queue import GPUTaskQueue, Task, TaskStatus


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


@patch("gpu_queue.MAX_OUTPUT_SIZE", 1000)  # 1KB for testing purposes
@patch("gpu_queue.BACKUP_COUNT", 2)
def test_rotating_file_handler_1000_lines(gpu_queue, setup_test_environment):
    run_rotating_file_handler_test(gpu_queue, setup_test_environment, 1000)


@patch("gpu_queue.MAX_OUTPUT_SIZE", 1000)  # 1KB for testing purposes
@patch("gpu_queue.BACKUP_COUNT", 2)
def test_rotating_file_handler_20_lines(gpu_queue, setup_test_environment):
    run_rotating_file_handler_test(gpu_queue, setup_test_environment, 20)


if __name__ == "__main__":
    pytest.main(
        "tests/test_gpu_queue.py::test_rotating_file_handler_1000_lines".split(
            " "
        )
    )
