#!/bin/sh
gunicorn api.main:create_app $GUNICORN_CMD_ARGS
