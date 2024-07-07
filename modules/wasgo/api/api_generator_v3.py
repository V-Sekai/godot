import json
import os
import re
from pathlib import Path


def find_files_recursively(dir, glob_pattern):
    return ["{0}".format(path.relative_to(dir)) for path in Path(dir).rglob(glob_pattern)]


def parse_method(method_line, class_name):
    if "(" not in method_line:
        return {}

    method = {}
    pre_paren = method_line[: method_line.find("(")]
    post_paren = method_line[method_line.find("(") :]
    pre_words = [word.strip() for word in pre_paren.split()]
    method["virtual"] = "virtual" in pre_words
    method["static"] = "static" in pre_words
    pre_word_names = [
        word for word in pre_words if word not in ["virtual", "static", "inline"]
    ]  # TODO add other qualifiers
    if not pre_word_names:
        return {}
    method["name"] = pre_word_names[-1]
    if len(pre_word_names) > 1:
        method["return_type"] = " ".join(pre_word_names[:-1])
        method["constructor"] = False
        method["destructor"] = False
    elif class_name in method["name"]:
        method["return_type"] = ""
        if method["name"][0] == "~":
            method["constructor"] = False
            method["destructor"] = True
        else:
            method["constructor"] = True
            method["destructor"] = False
    else:  # might have been a preprocessor macro or something
        return {}

    if ")" not in post_paren:
        return {}
    argument_string = post_paren[1 : post_paren.rfind(")")]
    if len(argument_string) < 2:
        method["arguments"] = []

    args = []
    arg_type = ""
    arg_name = ""
    arg_value = ""
    default_value = False
    reading_type = True
    template_depth = 0
    for character in argument_string:
        if character == "," and template_depth == 0:
            if arg_name:
                arg = {
                    "name": arg_name.strip(),
                    "type": arg_type.strip(),
                    "default_value": arg_value.strip(),
                    "has_default_value": default_value,
                }
                args.append(arg)
                arg_name = ""
                arg_type = ""
                arg_value = ""
                default_value = False
                reading_type = True
        elif character == "=":
            default_value = True
            reading_type = False
        elif default_value:
            arg_value += character
        elif character == "<":
            template_depth += 1
            if reading_type:
                arg_type += character
            else:
                arg_name += character
        elif character == ">":
            template_depth -= 1
            if reading_type:
                arg_type += character
            else:
                arg_name += character
        elif character == " " and reading_type and template_depth == 0:
            if arg_type:
                reading_type = False
        elif reading_type:
            arg_type += character
        else:
            arg_name += character
    if arg_name:
        arg = {
            "name": arg_name.strip(),
            "type": arg_type.strip(),
            "default_value": arg_value.strip(),
            "has_default_value": default_value,
        }
        args.append(arg)
    method["arguments"] = args
    return method


def extract_methods_from_header(header_content, class_name):
    method_pattern = re.compile(r"\b(?:virtual\s+|static\s+)?\w[\w\s:&*<>]*\s+\w+\s*\([^)]*\)\s*(?:const)?\s*;")
    methods = []
    for line in header_content.splitlines():
        line = line.strip()
        if method_pattern.match(line):
            method_info = parse_method(line, class_name)
            if method_info:
                methods.append(method_info)
    return methods


def parse_struct(struct_lines):
    return parse_class(struct_lines, True)


def parse_class(class_lines, public_default=False):
    if not class_lines:
        return {}
    class_dict = {"classes": [], "structs": [], "enums": [], "constants": [], "methods": []}
    public = public_default  # change to true for structs

    curly_brace_count = 0
    nested_class_lines = []
    nested_struct_lines = []
    enum_lines = []
    for index, line in enumerate(class_lines):
        if index == 0:
            # extract class names and parent classes from the first line
            class_name = line[5:]  # remove the class keyword
            parent_string = ""
            if "{" in class_name:
                class_name = class_name[: class_name.find("{")]
            if ":" in class_name:
                char_index = class_name.find(":")
                parent_string = class_name[char_index:]
                class_name = class_name[:char_index]
            # remove whitespaces
            class_dict["name"] = class_name.strip()
            # Check for multiple inheritance
            parent_words = [s.strip() for s in parent_string.split(",") if s]
            if len(parent_words) > 1:
                class_dict["inheritance"] = "multiple"
                class_dict["base_classes"] = parent_words
            elif len(parent_words) == 1:
                class_dict["inheritance"] = "single"
                class_dict["base_class"] = parent_words[0]
            else:
                class_dict["inheritance"] = ""
                class_dict["base_class"] = ""
        elif line.startswith("public:"):  # we assume that's gonna be the whole line because the input is prettyfied
            public = True
        elif public:
            if "{" in line:
                curly_brace_count += 1

            if nested_class_lines:
                nested_class_lines.append(line)
            elif nested_struct_lines:
                nested_struct_lines.append(line)
            elif enum_lines:
                enum_lines.append(line)
            else:
                if line.startswith("class"):
                    nested_class_lines.append(line)
                elif line.startswith("struct"):
                    nested_struct_lines.append(line)
                elif line.startswith("enum"):
                    enum_lines.append(line)
                elif "(" in line:
                    method = parse_method(line, class_dict["name"])
                    if method:
                        class_dict["methods"].append(method)

            if "}" in line:
                curly_brace_count -= 1
                if curly_brace_count == 0:
                    if nested_class_lines:
                        class_dict["classes"].append(parse_class(nested_class_lines))
                        nested_class_lines.clear()
                    if nested_struct_lines:
                        class_dict["structs"].append(parse_struct(nested_struct_lines))
                        nested_struct_lines.clear()
                    if enum_lines:
                        class_dict["enums"].append(enum_lines)  # just keep the whole thing without changing it
                        enum_lines.clear()

            if curly_brace_count == 0:
                if line.startswith("protected:") or line.startswith("private:"):
                    public = False

    # Check if singleton
    class_dict["singleton"] = any(m["name"] == "get_singleton" for m in class_dict["methods"])

    # Check if instanciable (i.e., has a public constructor)
    class_dict["instanciable"] = any(m["constructor"] and public for m in class_dict["methods"])

    return class_dict


def parse_header_file(file_path):
    # assumes that header files are prettified and have newlines at the regular places
    # if that's not the case, just run a prettifier on the files and try again
    # note the opening brace must be on the same line for class, struct, enum, and method definitions
    try:
        with open(file_path, "r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as input_file:
            lines = input_file.readlines()

    classes = []
    structs = []
    enums = []

    class_lines = []
    struct_lines = []
    enum_lines = []

    curly_brace_count = 0
    for raw_line in lines:
        line = raw_line.strip()
        if "{" in line:
            curly_brace_count += 1

        # continue the current definition
        if class_lines:
            class_lines.append(line)
        elif struct_lines:
            struct_lines.append(line)
        elif enum_lines:
            enum_lines.append(line)
        elif curly_brace_count == 1:  # else see if we should start a new definition
            if line.startswith("class"):
                class_lines.append(line)
            elif line.startswith("struct"):
                struct_lines.append(line)
            elif line.startswith("enum") and "{" in line:
                enum_lines.append(line)

        if "}" in line:
            curly_brace_count -= 1

        if curly_brace_count == 0:
            if class_lines:
                classes.append(parse_class(class_lines))
                class_lines.clear()
            elif struct_lines:
                structs.append(parse_struct(struct_lines))
                struct_lines.clear()
            elif enum_lines:
                enums.append(enum_lines)
                enum_lines.clear()

    header_file = {"classes": classes, "structs": structs, "enums": enums}
    return header_file


if __name__ == "__main__":
    input_folder = "../../../"
    output_folder = "./Generated"

    # Step 1 make a dictionary that shows the relationship of files to class to methods
    # Step 2 create the empty skeleton class that will live on the Wasm side. Our current target is C++, so these will be C++ style header and cpp files
    # Step 3 create the wasm side wrapper functions, these are all extern "C" functions which means they'll be exported in the global namespace with a _wasgo_ header. We'll keep these in a separate header file to not pollute the global namespace too much
    # Step 4 create the Godot side wrapper functions, these should have the same name and signature as their wasm counterparts, but they don't have to be extern "C". Unlike the wasm wrappers, these will have bodies that call Godot functions on behalf of wasm
    # Step 5 create the function table that links the Godot wrappers to the wasm wrappers

    # ******* STEP 1 ********
    input_files = find_files_recursively(input_folder, "*.h")

    file_classes_dict = {}
    for input_file in input_files:
        if not input_file.startswith("thirdparty/") and not input_file.startswith("tests/"):
            file_classes_dict[input_file] = parse_header_file(os.path.join(input_folder, input_file))

    with open("wasgo_api.json", "w") as file:
        file.write(json.dumps(file_classes_dict, indent=4))  # use `json.loads` to do the reverse
