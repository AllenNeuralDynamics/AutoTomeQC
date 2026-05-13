from nicegui import Event
from nicegui.events import UploadEventArguments, ClickEventArguments

# caller: components/loading_overlay.py
# subscribers: controllers/state_controller.py
is_running = Event[None]()
fetch_config = Event[None]()

# caller: components/uploader_sidebar.py
# subscribers: controllers/uploader_controller.py
on_upload = Event[UploadEventArguments]()
on_process = Event[ClickEventArguments]()
