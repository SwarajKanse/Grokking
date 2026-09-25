# AZURE_SETUP.md — Handing Azure off to Antigravity

Supersedes `AWS_SETUP.md` for now — the Azure Education credit ($9,555,
154 days, $0 spent) already exists and is already verified to your
account, so almost none of the AWS setup friction applies here.

## The one thing only you can do
Run `az login` once (install the Azure CLI first if it isn't already:
`winget install Microsoft.AzureCLI` on Windows, or the installer at
learn.microsoft.com/cli/azure/install-azure-cli). It opens a browser,
you sign in with the same account as the screenshot, done. This is the
only step that has to be interactive — after this, valid credentials are
cached locally and Antigravity's terminal can run `az` commands directly
using your existing session.

Everything else below is one prompt.

## VM sizing (verified pricing, East US)
| Size | vCPU / RAM | $/hr | Use |
|---|---|---|---|
| `Standard_D16s_v5` | 16 / 64 GiB | $0.768 | Default — Phase 3 feature engineering, Phase 4 model training |
| `Standard_E16s_v5` | 16 / 128 GiB | ~$1.10-1.20 (not independently priced here, same family pattern as D→E) | Step up to this only if `D16s_v5` hits memory pressure |

## The Antigravity prompt

> I've run `az login` — Azure CLI has a valid cached session now, use it
> directly. Set up and run our pipeline on an Azure VM instead of this
> laptop:
>
> 1. Create a resource group for this project (pick a clear name, one
>    region — East US has the cheapest published rate for the VM size
>    below).
> 2. Create one Linux VM: `Standard_D16s_v5` (step up to
>    `Standard_E16s_v5` only if we hit memory limits), Ubuntu, generate an
>    SSH key pair for it if one doesn't exist, and a network security
>    group allowing SSH only from my current IP — not open to the
>    internet.
> 3. Create a Storage Account + Blob container, upload `dataset/` to it,
>    confirm the upload.
> 4. SSH in, install Python/git/the packages in `requirements.txt`, pull
>    the dataset down from Blob Storage onto the VM's local disk — don't
>    read directly from Blob Storage in the hot path.
> 5. Run the pipeline phase I tell you to, over SSH, the exact same code
>    in this repo — nothing about moving to Azure changes `AGENTS.md` or
>    the phase discipline in `TECHNICAL_APPROACH.md`. Stream output back,
>    and copy log/report files into this repo's `experiments/` folder
>    afterward, same as a local run.
> 6. Set up an Azure Budget (`az consumption budget create`, or the
>    portal if that command doesn't exist in this CLI version — either is
>    fine) at $500 with an alert at 80%, just as a sanity check, not
>    because the credit pool is remotely at risk.
> 7. When I tell you to stop: **use `az vm deallocate`, not just an OS
>    shutdown.** Azure keeps billing compute for a VM that's merely
>    stopped-but-not-deallocated — only deallocation actually releases the
>    charge. Confirm the VM shows as "Stopped (deallocated)" before ending
>    the session, not just "Stopped."
>
> Before the first `az vm create` or `az group create` command, show me
> exactly what you're about to run so I can see it once.

## Same reminder as before, different service names
The "AWS managed AI services" boundary carries over exactly: using the VM
as a bigger computer is fine. Calling Azure AI Language, Azure Maps
(geocoding/address validation), or Azure AI Foundry / Azure OpenAI
(hosted foundation models) to help resolve entities crosses the same
external-lookup line the challenge rules draw — regardless of it being
Azure credit instead of AWS credit. Don't reach for these as a shortcut.
