import importlib.util
import json
import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.service import shell

app = FastAPI(title="GreenLit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    response = JSONResponse({"message": "hello world"})
    return response


# load API Routers
routes = [x.rstrip(".py") for x in os.listdir("api/route") if x.endswith(".py") and not x.startswith("_")]
with open("api/route/blacklist.json") as f:

    blacklist = json.load(f)

for route in routes:
    shell.print_cyan_message(f"Loading {route}...")
    if route in blacklist:
        shell.print_yellow_message("Blacklisted. Skipping...")
        continue
    try:
        importlib.util.spec_from_file_location(route, f"api/route/{route}.py")
        module = importlib.import_module(f"api.route.{route}")
        module.setup(app)
        shell.print_green_message("Success!")
    except Exception as e:
        shell.print_red_message(f"Failed:")
        print(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
