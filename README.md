# Interactive Hyperparameter Optimization in Multi-Objective Problems via Preference Learning

This is the repository of the paper ["Interactive Hyperparameter Optimization in Multi-Objective Problems via Preference Learning"](https://arxiv.org/abs/2309.03581).

## Requirements

In order to reproduce the experiments in any operating systems, Docker is required: [https://www.docker.com/](https://www.docker.com/).
Install it, and be sure that it is running when trying to reproduce the experiments.

To test if Docker is installed correctly:

- open the terminal;
- run ```docker run hello-world```.


***Expected output:***

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete
Digest: sha256:7d246653d0511db2a6b2e0436cfd0e52ac8c066000264b3ce63331ac66dca625
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## Reproducing the experiments

The instructions are valid for Unix-like systems (e.g., Linux Ubuntu, MacOS) and Windows (if using PowerShell).


Open the terminal and type:

```
docker run -it --volume ${PWD}/interactive-mo-ml:/home/interactive-mo-ml ghcr.io/josephgiovanelli/interactive-mo-ml:1.0.0
```

This creates and mounts the folder ```interactive-mo-ml``` into the container (which is populated with the code and the necessary scenarios), and run the paper experiments.

## Structure

The structure of the project is the follow:

- .github contains the material for the GitHub action;
- scripts contains the running scripts;
- src contains the source code;
- .gitattributes and .gitignore are configuration git files;
- Dockerfile is the configuration file to build the Docker container;
- README.md describes the content of the project;
- requirements.txt lists the required python packages.

The following folders will be generated
- input containing the configuration files to run the experiments;
- logs containing the logs of the experiments;
- output containing the results of the experiments;
- plots containing the tables and figures of the paper;
- smac3_output containing details of the performed optimizations.
