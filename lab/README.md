# Aegra Docker Multihost Lab

This lab is an isolated Docker Compose target range for Aegra. It creates a DMZ
network and an internal-only network. `aegra_agent` is attached only to the DMZ;
`pivot-ssh` is the only container attached to both networks.

## Topology

| Service | Address | Network | Purpose |
| --- | --- | --- | --- |
| `aegra_agent` | `10.20.0.10` | DMZ | Aegra API and tool runner |
| `dvwa` | `10.20.0.20` | DMZ | Vulnerable web target |
| `juice-shop` | `10.20.0.21` | DMZ | Vulnerable web target |
| `vulhub-s2-045` | `10.20.0.22` | DMZ | Struts2 S2-045 target |
| `pivot-ssh` | `10.20.0.30`, `10.30.0.30` | DMZ, internal | SSH jump host |
| `internal-web` | `10.30.0.40` | internal | Internal HTTP service |
| `internal-db` | `10.30.0.50` | internal | Internal PostgreSQL service |

## Start

```powershell
docker compose -f lab/docker-compose.yml build
docker compose -f lab/docker-compose.yml up -d
curl.exe --noproxy "*" http://127.0.0.1:8001/health
```

## Network Checks

DMZ discovery should work directly:

```powershell
docker compose -f lab/docker-compose.yml exec aegra_agent nmap -sV 10.20.0.0/24
```

Internal access should not work directly from `aegra_agent`:

```powershell
docker compose -f lab/docker-compose.yml exec aegra_agent nmap -Pn -p 80 10.30.0.40
```

Open a SOCKS proxy through the jump host:

```powershell
docker compose -f lab/docker-compose.yml exec -d aegra_agent sshpass -p pivotpass ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -N -D 127.0.0.1:1080 pivot@10.20.0.30
```

Then verify internal access through the pivot:

```powershell
docker compose -f lab/docker-compose.yml exec aegra_agent curl --proxy socks5h://127.0.0.1:1080 http://10.30.0.40
```

Runtime and audit output is persisted under `lab/outputs/`.

## Stop

```powershell
docker compose -f lab/docker-compose.yml down
```
