import logging
import threading
import random
from time import sleep
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KeyPool:
    """A pool of API keys for OpenAI's API. Support thread-safe operations"""

    def __init__(self, keys: List[str], verbose: bool = False):
        self.accounts = [{"key": k} for k in keys]
        for account in self.accounts:
            account["status"] = "ready"
        self.ready_num = threading.Semaphore(len(self.accounts))
        self.waiting_num = threading.Semaphore(0)
        self.blocked_num = threading.Semaphore(0)
        self.lock = threading.Lock()
        if verbose:
            threading.Thread(target=self.report_stats, daemon=True).start()

    def report_stats(self):
        while True:
            logger.debug(
                f"ready_num: {self.ready_num._value}, waiting_num:"
                f" {self.waiting_num._value}, blocked_num:"
                f" {self.blocked_num._value}"
            )
            sleep(2 * 60)

    def pop(self, get_account=False):
        """Get a key from the pool

        Args:
            get_account (bool, optional): If True, return the account dict. Defaults to False.
        """
        with self.lock:
            while True:
                random_list = list(range(len(self.accounts)))
                random.shuffle(random_list)
                for i in random_list:
                    account = self.accounts[i]
                    if account["status"] == "ready":
                        account["status"] = "waiting"
                        self.ready_num.acquire()  # self.ready_num -= 1
                        self.waiting_num.release()  # self.waiting_num += 1
                        if not get_account:
                            return account["key"]
                        else:
                            return account
                logger.debug("No key is ready. Retry after 5 seconds.")
                sleep(5)

    def free(self, key: str):
        """Free a key"""
        with self.lock:
            for account in self.accounts:
                if account["key"] == key:
                    if account["status"] == "waiting":
                        account["status"] = "ready"
                        self.ready_num.release()  # self.ready_num += 1
                        self.waiting_num.acquire()  # self.waiting_num -= 1
                    else:
                        logger.critical(
                            f"Key found for {key} but it's status is"
                            f" {account['status']} instead of waiting when"
                            " trying to free."
                        )
                    return
        logger.critical(f"Key not found for {key} when trying to free.")

    def unblock(self, key: str):
        """Unblock a key"""
        with self.lock:
            for account in self.accounts:
                if account["key"] == key:
                    if account["status"] == "blocked":
                        account["status"] = "ready"
                        self.blocked_num.acquire()  # self.blocked_num -= 1
                        self.ready_num.release()  # self.ready_num += 1
                    else:
                        logger.critical(
                            f"Key found for {key} but it's status is"
                            f" {account['status']} instead of blocked when"
                            " trying to unblock."
                        )
                    return
        logger.critical(f"Key not found for {key} when trying to unblock.")

    def block(self, key: str, duration_sec=5):
        """Block a key for a while"""
        with self.lock:
            for account in self.accounts:
                if account["key"] == key:
                    if account["status"] == "waiting":
                        account["status"] = "blocked"
                        self.blocked_num.release()  # self.blocked_num += 1
                        self.waiting_num.acquire()  # self.waiting_num -= 1
                        # unblock after duration_sec
                        threading.Timer(
                            duration_sec, self.unblock, args=[key]
                        ).start()
                    else:
                        logger.critical(
                            f"Key found for {key} but it's status is"
                            f" {account['status']} instead of waiting when"
                            " trying to block."
                        )
                    return
        logger.critical(f"Key not found for {key} when tyring to block.")
