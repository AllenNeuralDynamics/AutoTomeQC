from nicegui import Event

# components/loading_overlay.py calls 
is_running = Event[str]()
fetch_config = Event[str]()