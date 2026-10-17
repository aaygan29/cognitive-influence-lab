import re
def set_macro(s, name, value):
    # Replace the whole \newcommand{\name}{...} up to the last brace on its line.
    return re.sub(rf"(\\newcommand{{\\{name}}}{{)[^\n]*(}})\s*$",
                  lambda m: f"\\newcommand{{\\{name}}}{{{value}}}", s, flags=re.M)
