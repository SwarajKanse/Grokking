# Azure Education Subscription Capacity & VM Audit

- **Audit Timestamp:** 2026-09-25T13:35:00+05:30 (08:05:00 UTC)
- **Active Subscription:** Azure for Students
- **Subscription ID:** `072cd36c-ef96-498e-bf7d-a40b6dba47ae`
- **Tenant ID:** `2a214bae-08ef-4bb7-8926-1058c05005e6`
- **Audit Type:** Read-Only Discovery (No Azure resource created, resized, modified, or deleted)

---

## 1. Governance & Regional Constraints

### 1.1 Policy Restriction: `sys.regionrestriction`
An Azure Policy assignment (`/providers/Microsoft.Authorization/policyAssignments/sys.regionrestriction`) enforces a strict location whitelist on this subscription:
- **Allowed Regions:** `centralindia`, `austriaeast`, `malaysiawest`, `koreacentral`, `uaenorth`
- **Non-compliance Rule:** Any deployment attempted outside these 5 regions (including `eastus` or `eastus2`) is rejected at ARM preflight with:
  `RequestDisallowedByAzure: This policy maintains a set of best available regions where your subscription can deploy resources.`

### 1.2 Regional & Family Compute Quotas
Measured via `az vm list-usage` across all eligible regions:

| Quota Metric | Central India | Austria East | Malaysia West | Korea Central | UAE North | East US (Policy Blocked) |
|---|---|---|---|---|---|---|
| **Total Regional vCPUs** | **6 (Usage: 4)** | **6 (Usage: 0)** | **6 (Usage: 0)** | **6 (Usage: 0)** | **6 (Usage: 0)** | **6 (Usage: 0)** |
| **Standard ESv3 Family** | **4 (Usage: 4)** | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) |
| **Standard EASv4 Family** | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) |
| **Standard EDSv4 Family** | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) |
| **Standard DSv3 Family** | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) |
| **Standard DASv4 Family** | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) |
| **Standard BS Family** | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) | 4 (Usage: 0) |
| **Standard Basv2 / Bsv2** | 10 (Capped by 6) | 10 (Capped by 6) | 10 (Capped by 6) | 10 (Capped by 6) | 10 (Capped by 6) | 10 (Capped by 6) |
| **Standard DSv5 / ESv5** | **0** | **0** | **0** | **0** | **0** | **0** |
| **Standard M-Series** | **0** | **0** | **0** | **0** | **0** | **0** |

#### Quota Implication:
1. **8+ vCPU instances are impossible:** The global regional quota is capped at 6 vCPUs, and general-purpose/memory-optimized families are individually capped at 4 vCPUs. Any 8-vCPU or 16-vCPU instance (`Standard_D8s_v5`, `Standard_D16s_v5`, `Standard_E8s_v3`) fails immediately with `QuotaExceeded`.
2. **64 GiB RAM instances are impossible:** Standard Azure compute ratios provide up to 8 GiB RAM per core (E-series). With a 4-core family cap, the maximum achievable RAM under standard compute is **32 GiB**. Reaching 64 GiB would require an 8-vCPU instance (`E8s`), which exceeds the regional cap of 6 cores.
3. **v5 Generation is unavailable:** Both `Standard DSv5 Family` and `Standard ESv5 Family` have an approved limit of **0** across all regions in this student subscription.

---

## 2. Existing Cloud Infrastructure Audit

| Resource | Value | Evidence / Command |
|---|---|---|
| **Resource Group** | `rg-ber-centralindia` (`centralindia`) | `az group list` |
| **VM Name** | `vm-ber-worker` | `az vm list -d` |
| **Current Size** | `Standard_E4s_v3` (4 vCPUs, 32 GiB RAM) | `hardwareProfile.vmSize` |
| **Power State** | `VM running` (Provisioning State: `Succeeded`) | `instanceView.statuses` |
| **Public IP** | `104.211.90.212` (Dynamic SKU) | `az vm list -d` |
| **Private IP** | `10.0.0.4` | `az vm list -d` |
| **NSG Exposure** | `nsg-ber-ssh` strictly permits Port 22 from `42.106.241.175/32` & `42.106.208.48/32`. Priority 65500 `DenyAllInBound` blocks all other internet traffic. | `az network nsg list` |
| **Attached OS Disk** | 64 GB Premium SSD (`Premium_LRS`), `/dev/root` mounted (~61 GB free) | `storageProfile.osDisk` |
| **Attached Data Disks** | None (`[]`) | `storageProfile.dataDisks` |
| **Temporary Disk** | 63 GB on `/mnt` (`/dev/sdb1`, NVMe/SCSI ephemeral) | VM `df -h` |
| **Local Storage Total** | ~120 GB usable across OS and temporary mount | VM `df -h` |

---

## 3. Evaluation of Candidate VM SKUs

All SKUs in `centralindia` were audited against subscription restrictions (`NotAvailableForSubscription`), family quota limits, and pricing from the Azure Retail Prices API:

| Rank | SKU Name | vCPUs | RAM (GiB) | Max Data Disks | Quota Limit / Family | Retail Price ($/hr) | Suitability Analysis |
|:---:|---|:---:|:---:|:---:|---|:---:|---|
| **1 (Best Available)** | **`Standard_E4as_v4`** | 4 | **32.0** | 8 | 4 (`Standard EASv4`) | **$0.158** | **Top Choice.** AMD EPYC 7742 (up to 3.4 GHz boost, faster single-thread clock than v3), 32 GiB RAM, 42.3% cheaper than E4s_v3 ($0.158 vs $0.274). |
| **2 (Current VM)** | **`Standard_E4s_v3`** | 4 | **32.0** | 8 | 4 (`Standard ESv3`) | **$0.274** | **Viable Runner-Up.** Intel Xeon (Broadwell/Skylake, 2.6–3.0 GHz), 32 GiB RAM. Currently deployed and fully functional. |
| **3** | **`Standard_E4ds_v4`** | 4 | **32.0** | 8 | 4 (`Standard EDSv4`) | ~$0.252 | Intel Cascade Lake (2.8 GHz), 32 GiB RAM, includes 150 GB local NVMe scratch disk. More expensive than E4as_v4. |
| **4** | **`Standard_D4as_v4`** | 4 | 16.0 | 8 | 4 (`Standard DASv4`) | **$0.123** | Compute-optimized AMD, but only 16 GiB RAM. Below preferred 32 GiB working memory for 2.2M candidate indexing. |
| **5** | **`Standard_D4s_v3`** | 4 | 16.0 | 8 | 4 (`Standard DSv3`) | **$0.210** | Intel 4 vCPU, 16 GiB RAM. Inferior to E-series on memory and more expensive than D4as_v4. |
| **6** | **`Standard_B4ms`** | 4 | 16.0 | 8 | 4 (`Standard BS`) | **$0.179** | Burstable instance; performance throttles to baseline (90% of 1 core) once CPU credits deplete during prolonged ML training. Not recommended for batch pipeline. |
| **— (Blocked)** | **`Standard_D16s_v5`** | 16 | 64.0 | 32 | **0** (`Standard DSv5`) | $0.808 | **Rejected by Quota.** Requested in original brief, but v5 family limit is 0, and regional core limit is 6. |

---

## 4. Disk Storage Evaluation (Addressing the 256 GB Requirement)

The user priority states: *"Disk: at least 256 GB free local disk preferred for raw data, candidate artifacts, partitioned Parquet, and model outputs."*

### Current State:
- OS Disk: 64 GB Premium SSD (`vm-ber-worker_OsDisk_1_...`) with ~58 GB free space.
- Temporary Resource Disk: 63 GB mounted on `/mnt` with ~60 GB free space.
- Total free local disk space currently: **~118–120 GB**.

### Quota Capacity for Expansion:
Per `az vm list-usage`, storage quota in Central India is virtually unlimited:
- `Premium Storage Managed Disks`: 1 in use, **Limit: 50,000**.
- `StandardSSDStorageDisks`: 0 in use, **Limit: 50,000**.
- `PremiumV2TotalDiskSizeInGB`: 0 in use, **Limit: 1,048,576 GB (1 PB)**.

### Viable Options to reach 256+ GB Free Disk:
1. **Option A (Add a Dedicated 256 GB Data Disk — Recommended):**
   - Attach a 256 GB Managed Data Disk (`P15 Premium SSD` or `E15 Standard SSD`).
   - Cost: `E15 Standard SSD` is ~$0.021/hr ($15.36/month); `P15 Premium SSD` is ~$0.048/hr ($34.56/month).
   - Advantage: Completely independent of the OS disk; can be detached/attached without rebuilding software.
2. **Option B (Resize OS Disk to 256 GB):**
   - Deallocate VM, resize OS disk to 256 GB (`az disk update --size-gb 256`), and restart VM.
   - Filesystem automatically expands to 256 GB.
   - Cost: P15 Premium SSD rate (~$0.048/hr).
3. **Option C (Keep 120 GB Current Setup with Intermediate File Pruning):**
   - The entire 7-file raw dataset is 2.5 GB.
   - Pre-computed candidate pairs Parquet is ~200–400 MB.
   - Extracted features Parquet is ~1.5–2.5 GB.
   - LightGBM model binary is ~50 MB.
   - Total pipeline working footprint is under 15 GB, which fits easily inside the existing 120 GB available space without additional cost.

---

## 5. Audit of Deviations from Original Brief (`D16s_v5` / `East US`)

| Specification in `AZURE_SETUP.md` | Actual Deployed Specification | Factual Azure Restriction Requiring Deviation |
|---|---|---|
| **Region: `East US`** | **`Central India`** | Azure Policy `sys.regionrestriction` strictly forbids resources outside `[centralindia, austriaeast, malaysiawest, koreacentral, uaenorth]`. Attempts to deploy in `eastus` fail with `RequestDisallowedByAzure`. |
| **vCPU: 16 vCPUs** | **4 vCPUs** | Subscription regional quota cap is **6 total vCPUs**, and general-purpose family quota is **4 vCPUs**. A 16-core instance cannot be provisioned. |
| **RAM: 64 GiB** | **32 GiB** | 64 GiB instances require 8 or 16 cores in E-series/D-series. Since core quota is capped at 6, maximum achievable RAM is 32 GiB (E4 series, 8 GiB/core). |
| **Architecture: `v5` (D16s_v5)** | **`v3` (E4s_v3) / `v4` (E4as_v4)** | `Standard DSv5 Family` and `Standard ESv5 Family` have an approved limit of **0 cores** under the Azure for Students subscription. |

---

## 6. Clear Recommendations

1. **Memory & Compute Choice:**
   - **Current VM (`Standard_E4s_v3`, 4 vCPUs, 32 GiB RAM):** It is **not merely the first SKU that worked**; it represents the **exact ceiling** of allowable memory on this Azure for Students subscription (32 GiB).
   - **Optimization Candidate (`Standard_E4as_v4`, 4 vCPUs, 32 GiB RAM):** Runs on newer AMD EPYC 7742 architecture (higher boost clock) and costs **$0.158/hr vs $0.274/hr** (saving 42%). If the user approves resizing, switching to `Standard_E4as_v4` offers better single-thread performance at a lower price.
2. **Disk Choice:**
   - The current 120 GB total free disk is sufficient for the ~15 GB pipeline footprint. However, if the user strictly desires 256+ GB free disk as stated in the priority, attaching an `E15` 256 GB Standard SSD disk ($0.021/hr) or expanding the OS disk to 256 GB ($0.048/hr) trivially satisfies it.
3. **Pipeline Adaptation for 32 GiB RAM:**
   - As mandated by priority 1 (*"32 GB acceptable only if feature generation is streamed"*): The Phase 3 extractor ([`extract_features.py`](file:///d:/Grokking/code/business_entity_resolution/src/extract_features.py)) was explicitly built with chunked 50,000-pair streaming and memory downcasting (`float32`/`int16`/`int8`), consuming <3 GB of peak RSS during execution.

---

## 7. Required Human Approval Gate

Per instructions, **no resource-changing command has been executed**.
Awaiting human direction on which option to proceed with:

- **Choice A (Keep & Run):** Keep `vm-ber-worker` as `Standard_E4s_v3` (32 GiB RAM, 120 GB disk, $0.274/hr) and proceed directly with Phase 3/4 pipeline execution.
- **Choice B (Resize to AMD EPYC):** Deallocate and resize VM to `Standard_E4as_v4` (32 GiB RAM, faster 3.4 GHz boost, $0.158/hr).
- **Choice C (Expand Disk):** Attach a 256 GB Managed Data Disk or expand OS disk to 256 GB before running the pipeline.
