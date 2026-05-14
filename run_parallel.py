import argparse
import copy
import multiprocessing as mp
import os
import time

import numpy as np


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def run_with_device(server, device_id, config_path, config_name, overrides):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    os.environ["OMP_NUM_THREADS"] = "2"

    # Now import the main script
    if config_name == "online_rl":
        from run_online import run
    else:
        raise NotImplementedError

    args = {
        "config_path": config_path,
        "config_name": config_name,
        "overrides": overrides,
    }
    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="online_rl")
    parser.add_argument("--agent_config", type=str, default="simbaV2")
    parser.add_argument("--env_type", type=str, default="dmc_hard")
    parser.add_argument("--device_ids", default=[0], nargs="+")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_exp_per_device", type=int, default=1)
    parser.add_argument("--server", type=str, default="local")
    parser.add_argument("--group_name", type=str, default="test")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--overrides", action="append", default=[])

    args = vars(parser.parse_args())
    seeds = (np.arange(args.pop("num_seeds")) * 1000).tolist()
    device_ids = args.pop("device_ids")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    num_devices = len(device_ids)
    num_exp_per_device = args.pop("num_exp_per_device")
    pool_size = num_devices * num_exp_per_device

    # create configurations for child run
    experiments = []
    config_path = args.pop("config_path")
    config_name = args.pop("config_name")
    server = args.pop("server")
    group_name = args.pop("group_name")
    exp_name = args.pop("exp_name")
    agent_config = args.pop("agent_config")

    # import library after CUDA_VISIBLE_DEVICES operation
    # from scale_rl.envs.d4rl import D4RL_MUJOCO
    from scale_rl.envs.dmc import DMC_EASY_MEDIUM, DMC_HARD
    from scale_rl.envs.mujoco import MUJOCO_ALL

    env_type = args.pop("env_type")

    # online
    if env_type == "mujoco":
        envs = MUJOCO_ALL
        env_configs = [env_type] * len(envs)

    elif env_type == "dmc_em":
        envs = DMC_EASY_MEDIUM
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_hard":
        envs = DMC_HARD
        env_configs = ["dmc"] * len(envs)

    elif env_type == "all":
        envs = (
            MUJOCO_ALL
            + DMC_EASY_MEDIUM
            + DMC_HARD
        )
        env_configs = (
            ["mujoco"] * len(MUJOCO_ALL)
            + ["dmc"] * len(DMC_EASY_MEDIUM)
            + ["dmc"] * len(DMC_HARD)
        )

    else:
        raise NotImplementedError

    for seed in seeds:
        for idx, env_name in enumerate(envs):
            exp = copy.deepcopy(args)  # copy overriding arguments
            exp["config_path"] = config_path
            exp["config_name"] = config_name

            exp["overrides"].append("agent=" + agent_config)
            exp["overrides"].append("env=" + env_configs[idx])
            exp["overrides"].append("env.env_name=" + env_name)

            exp["overrides"].append("server=" + server)
            exp["overrides"].append("group_name=" + group_name)
            exp["overrides"].append("exp_name=" + exp_name)
            exp["overrides"].append("seed=" + str(seed))

            experiments.append(exp)
            print(exp)

    # run parallel experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method("spawn")
    available_gpus = device_ids
    process_dict = {gpu_id: [] for gpu_id in device_ids}

    for exp in experiments:
        wait = True
        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)

            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(
                        process_dict.items(), key=lambda x: len(x[1])
                    )
                    break

            time.sleep(10)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp["device_id"] = str(gpu_id)
        process = mp.Process(
            target=run_with_device,
            args=(
                server,
                exp["device_id"],
                exp["config_path"],
                exp["config_name"],
                exp["overrides"],
            ),
        )
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)
