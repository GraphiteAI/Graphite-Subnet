> [!WARNING]
> This document assumes you have gone through the proper basic installation steps. If you haven't, please refer to the [installation guide](./installation.md) before proceeding.

# Validator Guide 📜

Welcome to the Validator Guide! 🎉

This guide will teach you how to run your validator smoothly. It goes over the set-up steps for your validator for the Graphite subnet.

<hr>

## Table of Contents 📑

1. [Setup](#setup)
2. [Running a Synthetic Validator](#running-a-synthetic-validator)
3. [Running an Organic Validator](#running-an-organic-validator)

<hr>

<a id="setup"></a>

## Setup

> [!IMPORTANT]
We have a set of minimum hardware requirements for running a validator. See [min_compute.yml](../min_compute.yml).

_Compute requirements were estimated based on conservative projections of network traffic. Better/More computationally intensive algorithms are likely to benefit from faster compute._

1. Head to Weights and Biases and create an account. You can sign up [here](https://wandb.ai/site).

2. Head to the `User Settings` page. <br>
<p align="center">
  <img src="../static/wandb-settings.png" alt="WandB Settings">
</p>

3. Scroll down to `Danger Zone`. <br>
<p align="center">
  <img align="center" src="../static/danger-zone.png" alt="Danger Zone">
</p>

4. Click on `Reveal` and copy the API key.

5. Create a `.env` file in the root directory of the project and add the following:
```bash
WANDB_API_KEY=YOUR_API_KEY
```

<hr>

> [!TIP]
> Rather than connecting to the `finney` endpoint, it is recommended that you run a [local subtensor](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md).

<a id="synthetic-vali"></a>

## Running a Synthetic Validator 
Synthetic validators generate randomly generated graph problems by using a binned distribution to select the size of the problem (number of nodes), followed by sampling from a uniform distribution to populate either the coordinates or edges between the nodes. 

These problems are then sent to miners to challenge them. Running a synthetic validator is as simple as executing <ins>**one**</ins> of the following instructions:
<br> 

#### <ins>Running the Validator</ins>
```bash
python3 neurons/validator.py --netuid NETUID --subtensor.network NETWORK --wallet.name NAME --wallet.hotkey HOTKEY --logging.trace --axon.port PORT --organic False
```  

<br>

#### <ins>Running the Validator with auto-restart using PM2</ins>
Install **pm2** into your machine. Navigate to your Graphite-Subnet repo and verify that you are in the right python environment.


For auto restart, please use **pm2** to run the validator with the following command:
```
pm2 start neurons/validator.py --name auto_update_graphite_validator --interpreter python3 -- --netuid NETUID --subtensor.network NETWORK --wallet.name NAME --wallet.hotkey HOTKEY --logging.trace --axon.port PORT --organic False
```


Verify that your validator is running correctly using `pm2 status`. You should see that `auto_update_graphite_validator` is running. Further inspect the process logs using `pm2 logs auto_update_graphite_validator`.

<br>

#### <ins>Running the Validator with auto-update</ins>
Note that you should <ins>**not**</ins> run this command along with the auto-restart command as it might cause unexpected behavior as they both initialize pm2 processes of the same name. Install **pm2** and **jq** on your machine before running this code.



For auto update and auto restart, please run `run.sh` which runs a script that checks github for the current repository version every hour. If your local version differs, it pulls the new repository and installs it (The script is configured to run the pm2 process with the name: `auto_update_graphite_validator`):
```
source run.sh --netuid NETUID --subtensor.network NETWORK --wallet.name NAME --wallet.hotkey HOTKEY --logging.debug --axon.port PORT --organic False
```  
<br>  

_For the above commands, please replace:_ `NAME`, `NETWORK`, `HOTKEY`, `PORT` 
- Mainnet: `NETIUD 43`, `NETWORK finney or NETWORK <your_local_subtensor_endpoint>`
- Testnet: `NETUID 65`, `NETWORK test`

Edit the logging flag per your needs

<br>

<a id="organic-vali"></a>

## Running a Validator with your own site and backend
The backend automates proxy wallet creation, monitors validator performance, and ensures rebalancing actions are executed with precision. Designed for scalability and reliability, this service orchestrates the secure, transparent, and efficient operation of validators within a dynamic ecosystem. 


This guide explains how to set up and run the validator system, including proxy generation, configuration, and database schema. The base template is opensourced and can be found at the repository **yield_server**.

For database setup, refer to the [PostgreSQL 17 Documentation](https://www.postgresql.org/files/documentation/pdf/17/postgresql-17-A4.pdf).

For supabase database, refer to [Supabase Documentation](https://supabase.com/docs).

For logging, refer to [Logging Documentation](https://docs.python.org/3/library/logging.html).

Note that we will need to use both PostgreSQL and Supabase for the setup so do spend some time reading the above documents to get a gist of how to set them up.

---

## 🧠 Proxy Generation

Proxies are generated per activated leader (UUID-based), using `proxy_assignment_service.py`.  
Wallet names are derived from the shortuuid hash of the UUID.

---

## ⚙️ Environment Configuration (`.env.py`)

Ensure the following **12 environment variables** are set:

### Supabase
```env
SUPABASE_SERVICE_ROLE_KEY=<<from supabase>>
SUPABASE_URL=<<from supabase>>
SUPABASE_JWT_SECRET=<<from supabase>>
SUPABASE_JWT_ISSUER=<<from supabase>>
```

### Admin & Database
```env
ASYNC_DB_URL=<<PostgreSQL URL>>
ADMIN_KEY=<<Supabase admin user key>>
ADMIN_HASHKEY=<<128-byte unique key>>
```

### Wallet Paths
```env
LIVE_WALLET_PATH=<<wallets storage path>>
LIVE_MNEMONIC_FILEPATH=<<mnemonic archive path>>
LIVE_ARCHIVE_WALLET_PATH=<<archived wallets path>>
LIVE_ARCHIVE_MNEMONIC_FILEPATH=<<archived mnemonic text file path>>
LIVE_WALLET_PASSWORD=<<64-byte password>>
```

---

## 🧾 Constants

Refer to `yield_server/config/constant.py` for constants setup.

### Wallet & Email
```python
DEFAULT_EMAIL_TEMPLATE = "<<your superbase template>>"
SS58_ADDRESS_LENGTH = 48
```

### Balances
```python
MINIMUM_STARTING_BALANCE = 500_000_000       # 5 TAO
MAXIMUM_STARTING_BALANCE = 15_000_000_000    # 15 TAO
```

### Timing Intervals (in seconds)
```python
PROXY_GENERATION_INTERVAL = 60               # 1 min
LEADER_ACTIVATION_INTERVAL = 3600            # 1 hour
SUBNET_HOTKEY_SYNC_INTERVAL = 3600
SNAPSHOT_INTERVAL = 86400                    # 1 day
METRIC_UPDATE_INTERVAL = 3600
```

### Metrics
```python
VTRUST_THRESHOLD = 0.01
LOOKBACK_DAYS = 30
TOP_LEADERBOARD_SIZE = 10
SHARPE_RATIO_LOOKBACK_DAYS = 30
OTF_DAILY_APY = 13.89                        # Divide by 365 for daily %
MINIMUM_SAMPLES_FOR_SHARPE_RATIO = 30
DEFAULT_SHARPE_RATIO = 0
```

### Yield Signer Keys
```python
YIELD_SIGNER_HOTKEY = "<<base58 address>>"
YIELD_SIGNER_COLDKEY = "<<base58 address>>"
```

### Ports
```python
YIELD_VALIDATOR_REBALANCING_PORT = 6500
YIELD_VALIDATOR_PERFORMANCE_PORT = 6501
```

### Service Address
```python
YIELD_VALIDATOR_SERVICE_ADDRESS = "<<API endpoint>>"
```

### Rebalancing Params
```python
YIELD_REBALANCE_SLIPPAGE_TOLERANCE = 0.001
MAXIMUM_REBALANCE_RETRIES = 3
REBALANCE_COOLDOWN_PERIOD = 3600
DEFAULT_REBALANCE_MODE = "batch_all"
DEFAULT_REBALANCE_FEE = 0.001
DEFAULT_FEE_DESTINATION = "5HjMs5JDrLH3Hknmfm1gDq7nFYAv6M7t9v3EWMctSRXJS9HC"
DEFAULT_SLIPPAGE_TOLERANCE = 0.005
DEFAULT_MINIMUM_PROXY_BALANCE = 10_000_000   # 0.01 TAO
```

---

## 🗃️ Database Setup

Refer to `yield_server/backend/database/models.py` to define your database schema.

Initialize with:
```python
Base.metadata.create_all()
```

---

## Logging

Refer to `yield_server/config/logging.py` for logging setup.
This is the default logging configuraton offered by the bittensor subnet template. 
The levels are defined with a scoping logic. In other words, if you enable “info” level logging, then higher levels are also included (“warning”, “error”...).

✅ That’s it! Your validator and proxy setup should now be good to go.
