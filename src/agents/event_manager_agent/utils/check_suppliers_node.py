from state import SupplierState

def check_suppliers_node(state: SupplierState, config):
    # Logic to check suppliers and provide feedback
    if state.get("feedback"):
        return {"feedback": None}  # Reset feedback after processing
    return {"suppliers": state.get("suppliers", [])}