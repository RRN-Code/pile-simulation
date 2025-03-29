# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def get_deposit_pattern_truck(ore_grades, deposit_ratio_choice):
    """
    Returns a deposit pattern list for the truck process.
    For two ore grades:
        "1:1" returns ['A', 'B']
        "2:1" returns ['A', 'A', 'B']
    For three ore grades:
        "1:1:1" returns [lowest, middle, highest]
        "2:1:1" returns [lowest, lowest, middle, highest]
    """
    keys = list(ore_grades.keys())
    if len(keys) == 2:
        if deposit_ratio_choice == "1:1":
            return ['A', 'B']
        elif deposit_ratio_choice == "2:1":
            return ['A', 'A', 'B']
        else:
            return ['A', 'B']
    elif len(keys) == 3:
        # Sort keys in increasing order of ore grade value
        sorted_keys = ['A', 'B', 'C']
        if deposit_ratio_choice == "1:1:1":
            return sorted_keys
        elif deposit_ratio_choice == "2:1:1":
            return ['A', 'A', 'B', 'C']
        else:
            return sorted_keys
    else:
        # Default fallback
        return keys

