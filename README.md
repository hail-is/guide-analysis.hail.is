# GUIDE Browser
This repository contains the core code for the GUIDE browser: [https://guide-analysis.hail.is/](https://guide-analysis.hail.is/).

## Building
To build:
```sh
docker build . -t us-docker.pkg.dev/hail-vdc/hail/guide-browser:latest
docker push us-docker.pkg.dev/hail-vdc/hail/guide-browser:latest
```

## Data files
The hail team maintains copies of the data files for the browser in a requester
pays Google Cloud Storage bucket. To localize them for testing, run:
```sh
gcloud storage cp --billing-project=[PROJECT_ID] 'gs://guide-analysis-browser/data/*' .
tar xzvf all_phenos.tar.gz
```

## Running locally
This project uses [`uv`](https://docs.astral.sh/uv/) for project and dependency
management. After installing `uv`, and copying the data files into the local
directory the app can be tested in the following way:
```sh
uv sync
uv run shiny run app.py --port 8000
# then navigate to localhost:8000 in your browser
```

Testing the docker deployment can be done using `docker compose` like so:
```sh
docker compose up
# then navigate to localhost:8000 in a browser
```

## Deploy
The hail team maintains a configured server in Google Cloud that serves this
browser. No changes to nginx, systemd, or docker configuration should be
required to maintain the service as a whole.

To update the service, after logging into the server via `gcloud compute ssh`,
stop the guide-browser service, then pull the latest service changes and the
latest container image, update the data files if necessary, and finally, reload
systemd units and restart the guide-browser service:

```sh
sudo systemctl stop guide-analysis.service
cd /usr/src/guide-analysis.hail.is
git pull

docker pull us-docker.pkg.dev/hail-vdc/hail/guide-browser:latest

cd /var/www/guide-analysis
gcloud storage cp <DATA_FILES> .

sudo systemctl daemon-reload
sudo systemctl start guide-analysis.service
```

> NOTE:
> In order to pull the git repository as currently configured, you may need to
> enable ssh-agent forwarding. Github has [pretty good documentation] on how to
> do this. When connecting with `gcloud compute ssh`, you'll need to add `-- -A`
> to the command line to enable forwarding if it's not enabled by default in
> your ssh config.

[pretty good documentation]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding
