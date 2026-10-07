# Take-home task: message triage service

**Senior AI Systems Engineer, GAJAN Group Holdings**

Budget 3 hours. Send it back within 48 hours of receiving this.

Use any AI tools you like. We expect you to. Just tell us what you used and what you used it for.

## The situation

Our brands receive inbound customer messages across WhatsApp, email and Instagram. Today a human reads every one. At current volume that is already a bottleneck, and we are growing.

## What to build

A small service that reads each message in `messages.json` and, for each one, decides:

- what the customer wants
- what useful information can be pulled out of the message
- what should happen next, and who or what handles it

The shape of the output is your decision. Part of what we are assessing is whether you chose a structure that survives the messages in this file.

## Requirements

1. **It runs.** A README we can follow, with whatever setup is needed. Any language or stack.
2. **It scales to 10,000 messages a day.** Tell us the cost per 1,000 messages and how you arrived at the number. State your assumptions.
3. **Every result carries a confidence signal,** and you define the rule for when a human sees the message instead of the automation acting on it.
4. **Nothing in this file should be able to break it.** The file is real-shaped, not clean.

## What to send back

1. The code, as a repository link or a zip.
2. A README: how to run it, and anything we need to supply such as API keys.
3. A decision log, one page maximum:
   - what you chose not to build, and why
   - where this breaks, in your own assessment
   - what you would do with another day
   - which AI tools you used and for what

## How we assess this

We are not grading test coverage, polish or file structure. We are grading judgment: what you noticed in the data, what you decided to do about it, and whether your system behaves sensibly when something is not as expected.

The decision log is the first thing we read.

## Questions

Reply to this email. We answer within a few hours. Asking a good clarifying question is not a mark against you.
