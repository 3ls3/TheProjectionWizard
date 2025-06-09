run-docker:
	./scripts/bash/run_local_container.sh 

gcp-init:
	./scripts/bash/gcp_init.sh

push-image:
	./scripts/bash/push_image.sh

deploy:
	./scripts/bash/deploy_cloud_run.sh 