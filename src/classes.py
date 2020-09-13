def mapper(path, mapping_dict):
    if "basement" in path:
        mapping_dict[path] = "basement"
    elif "bathroom" in path:
        mapping_dict[path] = "bathroom"
    elif "bedroom" in path:
        mapping_dict[path] = "bedroom"
    elif "bookstore" in path:
        mapping_dict[path] = "bookstore"
    elif "cafe" in path:
        mapping_dict[path] = "cafe"
    elif "classroom" in path:
        mapping_dict[path] = "classroom"
    elif "conference_room" in path:
        mapping_dict[path] = "conference_room"
    elif "dining_room" in path:
        mapping_dict[path] = "dining_room"
    elif "furniture_store" in path:
        mapping_dict[path] = "furniture_store"
    elif "home_office" in path:
        mapping_dict[path] = "home_office"
    elif "home_storage" in path:
        mapping_dict[path] = "home_storage"
    elif "kitchen" in path:
        mapping_dict[path] = "kitchen"
    elif "living_room" in path:
        mapping_dict[path] = "living_room"
    elif "nyu_office" in path:
        mapping_dict[path] = "nyu_office"
    elif "office" in path:
        mapping_dict[path] = "office"
    elif "office_kitchen" in path:
        mapping_dict[path] = "office_kitchen"
    elif "playroom" in path:
        mapping_dict[path] = "playroom"
    elif "reception_room" in path:
        mapping_dict[path] = "reception_room"
    elif "student_lounge" in path:
        mapping_dict[path] = "student_lounge"
    elif "study" in path:
        mapping_dict[path] = "study"
    elif "study_room" in path:
        mapping_dict[path] = "study_room"
    else:
        mapping_dict[path] = "misc"
