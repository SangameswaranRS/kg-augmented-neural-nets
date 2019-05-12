import GlobalConfiguration, GlobalHelpers
from GlobalHelpers import get_absolute_path


# ---- Driver --- #
def mid2name(mid):
    # Translates mid to entity. Mappings obtained from wiki
    translation_file_path = get_absolute_path('/mid2wikipedia.tsv')
    translation_file = open(translation_file_path,'r')
    for line in translation_file:
        parsed_line = line.split('\t')
        if parsed_line[0] == mid:
            return parsed_line[1]
    return None
