from flask import Blueprint, Response

errors = Blueprint(name="errors", import_name=__name__)

@errors.errorhandler(401)
def custom_401(error):
    return Response("API Key required.", 401)