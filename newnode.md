# Steps to set up a new compute node

## Installing Linux
1. Wipe hard drive
2. Create a [bootable usb drive](https://ubuntu.com/tutorials/create-a-usb-stick-on-macos#5-etcher-configuration) with Ubuntu 24.04.3 LTS
3. Boot bios and choose usb drive as first option
4. Boot
5. Enable SSH
6. Create a user with username `cadmin` (please consult team for password)

## Network
1. Edit `/etc/hosts` and add `10.0.0.1 orca`. Should resemble the configuration provided in this repo under `/hosts/etchostscompute1.txt`
2. Use `sudo vim network.yaml` to create the network yaml file
    * Enter following information
        ```
        network:
        version: 2
        ethernets:
            eth0:
            dhcp4: no
            addresses:
                - 10.0.0.X/24 
        ```
    * Note file is space sensitive so, follow the format. `X` is specific to the computer (by convention, should be the number of the compute node). **Always check with the `/etc/hosts` file on `orca` (the head node) to make sure there does not already exist a compute node with that number.**
3. Use `esc` then `:wq` to exit vim
4. Enter `sudo netplan apply`
    * If you see any formatting errors, double check to make sure you followed the above format exactly

## Connecting to the Server
At this point, your node should be ready to connect to the server. Use an ethernet cable to connect your node to the network switch

### Verifying Connectivity
1. SSH into `cadmin` on the head node from your PC
2. Verify that you can `ping 10.0.0.X`, where `X` is the number of the added compute node 
3. Add `10.0.0.X computeX` to `/etc/hosts` on **orca**, the head node. 
4. Verify you can `ping computeX` from the head node
5. Verify that you can `ssh computeX`. 
    * If `ping computeX` works but `ssh computeX` fails, you likely set up the `cadmin` account wrong on the compute node


## Munge
Munge is an authentication client that Slurm uses to authenticate its communications

**Note that `sudo apt install` cannot be used on the compute nodes as they are not connected to the internet**

### Installing Munge
1. SSH into the compute node from the head node
2. Configure apt to use the head node's cache proxy
    * Create `/etc/apt/apt.conf.d/01proxy`:
        ```
        sudo tee /etc/apt/apt.conf.d/01proxy >/dev/null <<'EOF'
        Acquire::http::Proxy "http://10.0.0.1:3142";
        Acquire::https::Proxy "http://10.0.0.1:3142";
        EOF
        ```
3. Create the correct sources list
    * Note we are using `noble`
        ```
        . /etc/os-release
        echo $UBUNTU_CODENAME
        ```
4. Then overwrite `/etc/apt/sources.list` (NOTE: if this doesn’t work try manually replacing `${UBUNTU_CODENAME}` with `noble`):
     ```
        sudo tee /etc/apt/sources.list >/dev/null <<EOF
        deb http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME} main restricted universe multiverse
        deb http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-updates main restricted universe multiverse
        deb http://security.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-security main restricted universe multiverse
        deb http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-backports main restricted universe multiverse
        EOF
    ```
5. Remove any leftover bad list files:
    ```
    sudo find /etc/apt/sources.list.d -type f -name '*.list' -exec sudo sed -i 's|^deb http://10\.0\.0\.1|# &|' {} \;
    ```
6. Refresh and bring system to a clean state
    ```
    sudo apt-get clean
    sudo rm -rf /var/lib/apt/lists/*
    sudo apt-get update
    sudo apt --fix-broken install -y
    sudo apt-get dist-upgrade -y
    ```
7. Install Munge
    ```
    sudo apt-get install -y munge libmunge2
    ```
### Verify Installation
`which munge` should return `/usr/bin/munge`

Check`systemctl status munge`

## Chrony
Chrony is an NTP client, Munge requires all nodes to be time synchronized.

### Installation
1. Run `sudo apt-get install -y chrony`
2. Edit config file (`/etc/chrony/chrony.conf`)
    * Add `server 10.0.0.1 iburst`
3. Restart chrony: `sudo systemctl restart chrony`
4. Makestep: `sudo chronyc makestep` (allows chrony to make a large jump in time to correct for errors)
    * Should see `200 Ok`

### Verification
Run `chronyc tracking`, should see `Leap status: normal` at the bottom

### Restart munge
Run `sudo systemctl daemon-reload` \
Run `sudo systemctl restart munge`\
Verify with `sudo systemctl status munge`

## Slurm
**Steps marked with [CN] should be performed on the compute node, steps marked with [HN] should be performed on the head node**
1. [CN] Run `sudo apt-get install -y slurm-wlm`
2. [HN] Edit `slurm.conf` file
    * Add `NodeName=computeX NodeAddr=10.0.0.X State=UNKNOWN` where `X` is the number of your node
    * Push an updated version of `slurm.conf` to github
3. [CN] create `slurm.conf` file `sudo vim /etc/slurm/slurm.conf `
    * Copy and paste the contents from `slurm.conf` on the head node
4. [CN] Start slurm: `sudo systemctl restart slurmd`, verify with `sudo systemctl status slurmd`
5. [HN] Run `scontrol reconfigure` 
6. [HN] Verify status with `sinfo` on head node, should see the state of your node as `idle`

## NFS
### Mounting
1. Install NFS client: `sudo apt install nfs-common`
2. Edit `/etc/hosts` and add `10.0.0.2 compute1`
3. Add `compute1:/home /home nfs defaults 0 0` to `/etc/fstab`
4. Mount from compute1: `sudo mount /home`
### Verify
Type `ls`, should see `cadmin`'s home directory rather an empty directory
