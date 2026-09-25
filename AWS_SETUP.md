# AWS_SETUP.md — Handing AWS off to Antigravity

## The split
**You (≈10-15 minutes, console only, no AWS knowledge needed):** account
verification, one IAM user with a scoped policy, one SSH key pair, one
budget alert. These are the only steps that *have* to be a human clicking
a browser, because AWS account creation requires identity verification and
you never want to hand an autonomous agent your root password.

**Antigravity (everything else, one prompt):** launching the machine,
uploading data, installing dependencies, running every phase of the
pipeline, pulling results and logs back, stopping the machine, and keeping
an eye on spend.

---

## Part 1 — Your steps

### 1. Confirm you have a full AWS account
Your AWS Builder Center profile (required for registration) may not be the
same as a full AWS account with the $200 credit attached. If you haven't
already, sign up at aws.amazon.com — this needs an email/phone
verification step that can't be delegated.

### 2. Create one IAM user for Antigravity
Console → IAM → Users → Create user → **Attach policies directly** → skip
the AWS-managed policies → **Create policy** → paste the JSON below (edit
the bucket name first) → attach it to the new user.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2InstanceLifecycle",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances", "ec2:StartInstances", "ec2:StopInstances",
        "ec2:TerminateInstances", "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus", "ec2:CreateTags",
        "ec2:DescribeImages", "ec2:DescribeKeyPairs",
        "ec2:DescribeSecurityGroups", "ec2:DescribeSubnets",
        "ec2:DescribeVpcs", "ec2:AuthorizeSecurityGroupIngress",
        "ec2:CreateSecurityGroup"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3DataBucket",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject", "s3:GetObject", "s3:ListBucket",
        "s3:DeleteObject", "s3:CreateBucket"
      ],
      "Resource": [
        "arn:aws:s3:::REPLACE-WITH-YOUR-BUCKET-NAME",
        "arn:aws:s3:::REPLACE-WITH-YOUR-BUCKET-NAME/*"
      ]
    },
    {
      "Sid": "BudgetVisibility",
      "Effect": "Allow",
      "Action": ["budgets:ViewBudget", "ce:GetCostAndUsage"],
      "Resource": "*"
    }
  ]
}
```

This is deliberately narrow — EC2 lifecycle, one S3 bucket, budget
read-only. No IAM permissions, no ability to touch other services, nothing
close to full admin access. After creating the user, go to **Security
credentials → Create access key** and save the Access Key ID and Secret
Access Key somewhere you'll paste from once, not into any file that could
end up in git.

### 3. Create an SSH key pair
Console → EC2 → Key Pairs → Create key pair → download the `.pem` file.
This is what Antigravity will use to reach the machine. Keep it out of the
repo (add `*.pem` to `.gitignore` now, before it's a problem).

### 4. Set a budget alert
Console → Billing → Budgets → Create budget → Cost budget → $180 (leaving
headroom under the $200) → alerts at 50%, 80%, 100% to your email. This is
the simple, reliable version. AWS Budgets can also auto-apply a
spend-blocking policy at 100% — worth knowing it exists, but for a 3-day
window at roughly $1/hour, the realistic downside of forgetting to stop an
instance overnight is a few dollars, not something worth spending your
scarce setup time engineering around. The alert plus actually stopping the
instance when you're not using it is enough.

---

## Part 2 — Hand this to Antigravity

Paste the Access Key ID, Secret Access Key, the `.pem` file's path, and
your chosen bucket name into the prompt below.

> I have AWS credentials for a scoped IAM user (EC2 + one S3 bucket only)
> and an SSH key pair. Set up and run our pipeline on EC2 instead of this
> laptop:
>
> 1. Configure the AWS CLI locally with the credentials I'm giving you
>    (`aws configure` or equivalent) — don't write the secret key into any
>    file in this repo.
> 2. Launch one EC2 instance: `m5.4xlarge` to start (16 vCPU, 64 GiB —
>    step up to `r5.4xlarge` only if we hit memory limits), Ubuntu, the
>    key pair I gave you, a security group allowing SSH only from my
>    current IP (not 0.0.0.0/0).
> 3. Create the S3 bucket if it doesn't exist, upload `dataset/` to it,
>    and confirm the upload.
> 4. SSH in, install Python/git/the packages in `requirements.txt`, and
>    pull the dataset down from S3 onto the instance's local disk — don't
>    read directly from S3 in the hot path, it's slower than local disk
>    for repeated access.
> 5. Run the pipeline phase I tell you to, over SSH, exactly the same
>    code that's in this repo — nothing about moving to EC2 changes the
>    rules in `AGENTS.md` or the phase discipline in
>    `TECHNICAL_APPROACH.md`. Stream the output back so I can see it, and
>    copy any log/report files back into this repo's `experiments/`
>    folder afterward, the same as a local run would.
> 6. When I tell you to stop, stop (not terminate) the instance and
>    confirm it's actually stopped before ending the session.
>
> Before running the first `aws ec2 run-instances` command, show me
> exactly what you're about to run so I can check it — this is real money
> and my first time using AWS, so I want to see the first one before it's
> automatic.

That last line matters — for the very first cloud-provisioning command or
two, actually look at what it's about to run before approving, the same
way you'd review the first SQL migration an agent proposes against a real
database. After that pattern is established, it's fine to let it move
faster.

## One more reminder from AGENTS.md, restated because it matters more here
Moving to EC2 doesn't change the "no external databases, APIs, or lookups"
rule. It's fine to use AWS as a bigger computer. It is not fine to reach
for AWS Comprehend, Location Service, Textract, Bedrock-hosted models, or
anything similar that would supply outside knowledge about these
businesses — that crosses the same line a third-party API would,
regardless of whose credits are paying for it.
