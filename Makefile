run-docker:
	./scripts/run_local_container.sh 

gcp-init:
	./scripts/gcp_init.sh

push-image:
	./scripts/push_image.sh

deploy:
	./scripts/deploy_cloud_run.sh 