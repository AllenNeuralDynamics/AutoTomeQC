from nicegui import Event

# caller: components/loading_overlay.py
# subscribers: controllers/state_controller.py
is_running = Event[None]()
fetch_config = Event[None]()
