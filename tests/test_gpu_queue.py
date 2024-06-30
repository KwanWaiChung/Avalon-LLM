import pytest
from unittest.mock import patch, MagicMock
from gpu_queue import GPUTaskQueue, Task


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
    gpu_queue.add_task(command, 1, "output.txt", priority)
    assert len(gpu_queue.queue) == 1
    task = gpu_queue.queue[0]
    assert isinstance(task, Task)
    assert task.priority == 5
    assert task.command == command


def test_run_tasks_with_highest_priority(gpu_queue, mock_gpustat):
    with patch.object(gpu_queue, "_run_task") as mock_run:
        gpu_queue.add_task("echo test1", 1, "output1.txt", 1)
        gpu_queue.add_task("echo test2", 3, "output2.txt", 3)
        gpu_queue.add_task("echo test3", 1, "output3.txt", 5)

        gpu_queue.run_tasks()

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0].command == "echo test3"


def test_run_tasks_without_available_task_with_highest_priority(
    gpu_queue, mock_gpustat
):
    with patch.object(gpu_queue, "_run_task") as mock_run:
        gpu_queue.add_task("echo test1", 3, "output1.txt", 5)
        gpu_queue.add_task("echo test2", 1, "output2.txt", 3)
        gpu_queue.add_task("echo test3", 3, "output3.txt", 5)

        gpu_queue.run_tasks()
        assert mock_run.call_count == 0


@patch("subprocess.Popen")
def test_monitor_and_clean(mock_popen, gpu_queue):
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    task = Task(
        command="echo test", n_gpus=1, output_file="output.txt", priority=5
    )
    gpu_queue._run_task(task, [0])

    gpu_queue.monitor_and_clean()
    assert len(gpu_queue.running_processes) == 0


# if __name__ == "__main__":
#     pytest.main()
