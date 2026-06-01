# One-click AWS run for `genetic-algorithm.cpp`

## Prereqs (one-time)

1. Install AWS CLI: `brew install awscli` then `aws configure`.
2. Create EC2 key pair (download `.pem`):
   ```
   aws ec2 create-key-pair --key-name ga-key --region eu-central-1 \
     --query 'KeyMaterial' --output text > ~/.ssh/ga-key.pem
   chmod 400 ~/.ssh/ga-key.pem
   ```
3. Create security group allowing SSH from your IP:
   ```
   aws ec2 create-security-group --group-name ga-sg \
     --description "GA SSH" --region eu-central-1
   # then authorize-security-group-ingress for port 22 from your IP
   ```
   Note the `sg-xxxx` ID.
4. Pick a globally unique S3 bucket name (script creates it).

## Configure

Edit env vars at top of `launch.sh` or export them:

```bash
export REGION=eu-central-1
export KEY_NAME=ga-key
export SECURITY_GROUP_ID=sg-xxxxxxxx
export S3_BUCKET=my-unique-ga-bucket
```

## Run

```bash
chmod +x aws/*.sh
./aws/launch.sh
```

Defaults: `r7iz.2xlarge`, POP=20000, GEN=100, ELITE=200, RUNS=3.

Override per-run:

```bash
INSTANCE_TYPE=r7iz.8xlarge POPULATION_SIZE=80000 ELITE_COUNT=800 ./aws/launch.sh
```

Instance auto-terminates after run. No idle billing.

## Sizing cheat-sheet (1000x1000, zram enabled → matches macOS behavior)

| POP    | Instance        | RAM    | $/hr (on-demand) |
|--------|-----------------|--------|------------------|
| 20k    | r7iz.2xlarge    | 64 GB  | ~$0.74           |
| 40k    | r7iz.2xlarge    | 64 GB  | ~$0.74           |
| 80k    | r7iz.4xlarge    | 128 GB | ~$1.49           |
| 160k   | r7iz.8xlarge    | 256 GB | ~$2.97           |

Always measure first with `/usr/bin/time -v ./ga` (already wired into user-data, ends up in `run.log`).

## Fetch results

```bash
./aws/fetch-results.sh <RUN_ID>
```

`RUN_ID` is printed by `launch.sh`. Files land in `rezultati-1000x1000/`.

## Monitor while running

```bash
# Console output
aws ec2 get-console-output --instance-id <id> --region $REGION --output text

# Or SSH
ssh -i ~/.ssh/ga-key.pem ubuntu@<public-ip>
tail -f /var/log/ga-run.log
```

## Cost guard

- Instance has `shutdown-behavior=terminate`. End of `user-data.sh` calls `shutdown -h now`.
- If something hangs: `aws ec2 terminate-instances --instance-ids <id> --region $REGION`.

## Spot instances (60-70% cheaper)

Add to `launch.sh` `run-instances` call:
```
--instance-market-options 'MarketType=spot'
```
Risk: AWS can reclaim mid-run. OK for short runs.
