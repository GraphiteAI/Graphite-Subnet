# Bittensor Installation Guide 🚀

Welcome to the basic installation guide! This document will walk you through the steps to install the files you need on your local machine to get you ready to [mine](./miner.md) or [validate](./validator.md) on Graphite, right from scratch.

<hr>

## Table of Contents 📑

1. [Requirements](#requirements) 📋
2. [Installation](#installation) 🔧
   1. [Clone the repository](#clone-the-repository) 📂
   2. [Enter the directory](#enter-the-directory) ➡️
   3. [Set up a virtual environment (optional)](#set-up-a-virtual-environment) 🌐
   4. [Install PM2 (optional)](#install-pm2) 📦
   5. [Install the dependencies](#install-the-dependencies) 📦
   6. [Create your wallets](#create-your-wallets) 👝
   7. [Regenerate your wallets](#regenerate-your-wallets) 🔄
3. [Post-Installation Verification](#post-installation-verification) ✅
4. [Registration](#registration) 📝
5. [Troubleshooting](#troubleshooting) 🛠️

<a id="requirements"></a>

## Requirements 📋

Before installing Bittensor, ensure you have the following requirements:

- Python 3.10 or 3.11 🐍
- `pip` (Python package installer) 📦
- `git` 🛠️
- Met all other requirements specified in [min_compute.yml](../min_compute.yml) 📑

<hr>

<a id="installation"></a>

## Installation 🔧

<a id="clone-the-repository"></a>

### Clone the repository 📂

1. Clone the repository into your local machine by running the following command:

```sh
git clone https://github.com/GraphiteAI/Graphite-Subnet.git
```

<a id="enter-the-directory"></a>

### Enter the directory ➡️

2. Navigate into the directory you just cloned by running:

```sh
cd Graphite-Subnet
```

<a id="set-up-a-virtual-environment"></a>

### Set up a virtual environment (optional) 🌐

> [!TIP]
> Creating a virtual environment is highly recommended to avoid conflicts with other Python projects.

3. Create a virtual environment by running one of the following commands with your preferred Python version:

```sh
python3.10 -m venv <your_environment_name>
```

```sh
python3.11 -m venv <your_environment_name>
```

4. Activate the virtual environment by running:

```sh
source <your_environment_name>/bin/activate
```

<a id="install-pm2"></a>

### Install PM2 (optional) 📦

> [!TIP]
> PM2 is a process manager for Node.js applications. It allows you to keep your application alive forever, to reload it without downtime, and to manage application logs.

5. Install PM2 by running:

<strong>Linux:</strong>
```sh
sudo apt update && sudo apt install npm && sudo npm install pm2 -g && pm2 update
```

<strong>MacOS:</strong>
```sh
brew update && brew install npm && sudo npm install pm2 -g && pm2 update
```

6. Verify your installation by running:

```sh
pm2 --version
```

<a id="install-the-dependencies"></a>

### Install the dependencies 📦

7. Disable the `pip` cache:

```sh
export PIP_NO_CACHE_DIR=1
```

8. Install the dependencies by running:

```sh
pip install -r requirements.txt
```

9. Create a local and editable installation:

```sh
pip install -e .
```

<a id="create-your-wallets"></a>

### Create your wallets 👝

> [!NOTE]  
> For those that already have wallets, and are unsure on how to regenerate them, skip this section and head <a href="#regenerate-your-wallets">here</a> instead. <br>
> If your wallets are all set up, you can proceed to [Post-Installation Verification](#post-installation-verification).


10. To create a new coldkey:

```sh
btcli wallet new_coldkey --wallet.name <your_wallet_name>
```

11. To create a new hotkey:

```sh
btcli wallet new_hotkey --wallet.name <your_existing_coldkey_name> --wallet.hotkey <your_hotkey_name>
```

An example of creating a new coldkey and hotkey would be:

```sh
btcli wallet new_coldkey --wallet.name coldkey1
```

```sh
btcli wallet new_hotkey --wallet.name coldkey1 --wallet.hotkey hotkey1
```

<a id="regenerate-your-wallets"></a>

### Regenerate your wallets 🔄

> [!NOTE]  
> For those that just created their wallets in the previous step, ignore this section and proceed <a href="#post-installation">here</a> instead.

12. Regenerate your coldkey:

```sh
btcli wallet regen_coldkey --wallet.name <your_wallet_name> --mnemonic <your_mnemonic>
```

13. Regenerate your hotkey:

```sh
btcli wallet regen_hotkey --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey_name> --mnemonic <your_mnemonic>
```

<hr>

<a id="post-installation-verification"></a>

## Post-Installation Verification ✅

To verify that your wallets are created, you can run:

```sh
btcli wallet list
```

If you receive a prompt for the wallets path, it is most likely in the default path, so just enter the default input by pressing `Enter` or `Return`.

In any case, if you have it stored in a separate path, do specify the path when prompted or with the `--wallet.path` flag.

If installed correctly, this should display all the coldkeys and their respective hotkeys.

It should look something like this:

```sh
Wallets
├──
│   guide_validator (5DtJYgMRevSFVdavGqoYrM4P6Bbk1ifoiwop6mt1ikX6Y5Nq)
│   └── hotkey (5FeVAsVkzgmYsHLeRahUh1PUZwKCYdyBCSWH1YTFfRw7UV3y)
└──
    guide_miner (5FC8U6TSX2C5HKbk1kRizgPFAWJLiRKJnycXPUQv7MZzmn9T)
    └── hotkey (5EhoFzbSgnAK5KGt6Zs44v737wdzoya58frzNnqvP6abQ4bF)
```

<hr>

<a id="registration"></a>

## Registration 📝

> [!NOTE]
> Ensure that you have sufficient funds in your wallet to register your wallets onto the subnet.
> To receive testnet TAO, you can request some from the Community Moderators in the <a href="https://discord.gg/N65us8J5">Bittensor Discord</a>.

> [!TIP]
> Graphite is Subnet 43 on the mainnet and Subnet 65 on the testnet.

To register your wallets onto the subnet, you can run:

<strong>Mainnet</strong>
```sh
btcli subnet register --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey_name> --netuid 43
```

<strong>Testnet</strong>
```sh
btcli subnet register --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey_name> --netuid 65 --subtensor.network test
```

<hr>

<a id="troubleshooting"></a>

## Troubleshooting 🛠️

If you encounter issues during installation:

- Ensure your Python and `pip` versions are up to date.
- Create a virtual environment with the suitable Python versions to avoid conflicts with other Python projects.
- Reach out to the Graphite team on Discord for further assistance.

### With that, you're all set! 🚀

Now that you're all equipped with the necessary tools, you can choose to either:

- [Run a Validator](./validator.md)
- [Run a Miner](./miner.md)
