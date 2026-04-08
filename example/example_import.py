# example/example_import.py
from autotomeqc.core.autotome_service import AutoTomeService

def print_result(result):
    print(f"Processing: {result['filename']}")
    print(f"Status:     {result['qc_summary']}")
    if result['qc_summary'] == "FAIL":
        print(f"Reason:     {result.get('fail_reason', 'N/A')}")

    sections = result.get('sections', [])
    for i, data in enumerate(sections):
        qc_status = data.get('qc_result', 'UNKNOWN')
        area = data.get('area_in_pixels', 0)
        print(f" -> Section {i}: {qc_status} | Area: {area}px")
        criteria = data.get('criteria', {})
        for crit, res in criteria.items():
            icon = "✅" if res.get('pass_status') else "❌"
            label = res.get('label', 'N/A')
            print(f"    {icon} {crit}: {label}")
    print("-" * 40)

def main():
    service = AutoTomeService()
    if not service.start():
        return
    print("Service Ready! Starting processing...")
    try:
        for i in range(18):
            path = f"example/input_images/img{i}.jpg"
            print(f"\nProcessing: {path}")

            # ---------------------------------------------------------
            # WAY 1: Pass by File Path (Service loads the file)
            # ---------------------------------------------------------
            future = service.process(img_path=path)

            # ---------------------------------------------------------
            # WAY 2: Pass by Raw Frame (You load the file)
            # ---------------------------------------------------------
            #frame = cv2.imread(path)
            #if frame is None: continue 
            #future = service.process(frame=frame)
            # ---------------------------------------------------------

            # Print output
            result = future.result()
            print_result(result)

    except KeyboardInterrupt:
        pass
    finally:
        service.stop()

if __name__ == "__main__":
    main()