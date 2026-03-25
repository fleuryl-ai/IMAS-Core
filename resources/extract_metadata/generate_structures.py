import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse
import os

# Mapping IMAS data types to C++ types
imas_to_cpp_type = {
    "FLT_0D": "double*",
    "FLT_1D": "double*",
    "FLT_2D": "double*",
    "FLT_3D": "double*",
    "FLT_4D": "double*",
    "FLT_5D": "double*",
    "FLT_6D": "double*",
    "INT_0D": "int*",
    "INT_1D": "int*",
    "INT_2D": "int*",
    "INT_3D": "int*",
    "INT_4D": "int*",
    "INT_5D": "int*",
    "INT_6D": "int*",
    "CPX_0D": "std::complex<double>*",
    "CPX_1D": "std::complex<double>*",
    "CPX_2D": "std::complex<double>*",
    "CPX_3D": "std::complex<double>*",
    "CPX_4D": "std::complex<double>*",
    "CPX_5D": "std::complex<double>*",
    "CPX_6D": "std::complex<double>*",
    "STR_0D": "std::string",
    "STR_1D": "std::vector<std::string>",
    "struct_array": "std::vector",
    "structure": "",
}

# Default values for initialization
DEFAULT_INT_VALUE = -999999
DEFAULT_DOUBLE_VALUE = -9e40
DEFAULT_STRING_VALUE = ""

# Classes to represent fields and structures
class Field:
    def __init__(self, name: str, type_name: str, is_array: bool = False, is_vector: bool = False, is_struct: bool = False, original_name: str = None, rank: int = 0, timebasepath: str = ""):
        self.name = name
        self.type_name = type_name
        self.is_array = is_array
        self.is_vector = is_vector
        self.is_struct = is_struct
        self.extents_name = f"{name}_extents" if is_array else None
        self.original_name = original_name or name
        self.rank = rank
        self.timebasepath = timebasepath

class Structure:
    def __init__(self, name: str, fields: list):
        self.name = name
        self.fields = fields

def get_dimension_from_type(data_type):
    """Extract the dimension from the data type (e.g., INT_2D -> 2, FLT_3D -> 3)."""
    if "_" in data_type:
        dim_str = data_type.split("_")[-1].replace("D", "")
        try:
            dim = int(dim_str)
            if 0 <= dim <= 6:
                return dim
        except ValueError:
            pass
    return 0

def generate_cpp_type(field, ids_name):
    """Generate C++ type for a field based on its data_type."""
    data_type = field.get("data_type")
    field_name = field.get("name")
    original_name = field.get("original_name", field_name)
    
    if data_type == "struct_array":
        struct_name = "std::vector<" + original_name + "_struct" + ">"
        return struct_name
    elif data_type == "structure":
        return original_name
    elif data_type in imas_to_cpp_type:
        return imas_to_cpp_type[data_type]
    else:
        print(f"Warning: Unknown data type '{data_type}' for field '{field_name}' in {ids_name}")
        return "void*"

def get_field_initializer(cpp_type, field_name, data_type=None):
    """Return the appropriate initializer for a field based on its C++ type and data_type."""
    if cpp_type == "double":
        return f" = {DEFAULT_DOUBLE_VALUE}"
    elif cpp_type == "int":
        return f" = {DEFAULT_INT_VALUE}"
    elif cpp_type == "std::string":
        return f' = "{DEFAULT_STRING_VALUE}"'
    elif cpp_type.endswith("*"):
        return " = nullptr"
    return ""

def topological_sort(dependencies):
    """Perform topological sort on structures based on dependencies."""
    in_degree = {}
    graph = defaultdict(list)
    
    # Collect all structures (keys and dependencies)
    all_structures = set(dependencies.keys())
    for struct, deps in dependencies.items():
        all_structures.update(deps)
    
    # Initialize in-degrees
    for struct in all_structures:
        in_degree[struct] = 0
    
    # Build the graph and calculate in-degrees
    for struct, deps in dependencies.items():
        for dep in deps:
            graph[dep].append(struct)
            in_degree[struct] = in_degree.get(struct, 0) + 1
    
    # Initialize queue with nodes that have no incoming dependencies
    queue = [struct for struct, degree in in_degree.items() if degree == 0]
    sorted_order = []
    
    while queue:
        current = queue.pop(0)
        sorted_order.append(current)
        
        # Update in-degrees of neighbors
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for circular dependencies
    if len(sorted_order) != len(in_degree):
        print("Warning: Circular dependency detected in structure definitions")
        return list(all_structures)
    
    print(f"Debug: Topological sort order: {sorted_order}")
    return sorted_order

def pre_collect_structure_names(xml_element, structure_names):
    """Pre-collect all structure names from the XML to ensure they are known."""
    for field in xml_element.findall("field"):
        field_name = field.get("name")
        data_type = field.get("data_type")
        if field_name and data_type in ["structure", "struct_array"]:
            structure_names.add(field_name)
            pre_collect_structure_names(field, structure_names)

def collect_fields(xml_element, ids_name, structure_name, structures_list, parent_structure=None, excludes=None):
    """Recursively collect fields and substructures from an XML element, excluding fields matching any pattern in excludes."""
    if excludes is None:
        excludes = []
    fields = []
    seen_fields = set()

    for field in xml_element.findall("field"):
        field_name = field.get("name")
        if not field_name:
            print(f"Warning: Skipping unnamed field in structure '{structure_name}'")
            continue
        
        # Skip fields whose names contain any of the exclude patterns
        if any(exclude in field_name for exclude in excludes):
            print(f"Debug: Excluding field '{field_name}' in structure '{structure_name}' due to exclude pattern match")
            continue
        
        data_type = field.get("data_type")
        timebasepath = field.get("timebasepath", "")
        if data_type == "STR_1D": #TODO 
            continue
        original_name = field_name
        cpp_field_name = field_name
        if data_type == "struct_array":
            cpp_field_name = f"{field_name}_vect"
        
        cpp_type = generate_cpp_type({"name": cpp_field_name, "data_type": data_type, "original_name": original_name}, ids_name)
        is_array = data_type in ["FLT_1D", "FLT_2D", "FLT_3D", "FLT_4D", "FLT_5D", "FLT_6D",
                                 "INT_1D", "INT_2D", "INT_3D", "INT_4D", "INT_5D", "INT_6D",
                                 "CPX_1D", "CPX_2D", "CPX_3D", "CPX_4D", "CPX_5D", "CPX_6D"]
        is_vector = data_type == "struct_array"
        is_struct = data_type in ["structure", "struct_array"]
        rank = get_dimension_from_type(data_type)
        
        field_key = (cpp_field_name, cpp_type)
        if field_key not in seen_fields:
            fields.append(Field(cpp_field_name, cpp_type, is_array, is_vector, is_struct, original_name, rank, timebasepath))
            seen_fields.add(field_key)
        
        if data_type in ["structure", "struct_array"]:
            substructure_name = original_name
            existing_struct = next((s for s in structures_list if s.name == substructure_name), None)
            if existing_struct:
                new_fields = collect_fields(field, ids_name, substructure_name, structures_list, existing_struct, excludes)
                existing_seen_fields = {(f.name, f.type_name) for f in existing_struct.fields}
                for new_field in new_fields:
                    if (new_field.name, new_field.type_name) not in existing_seen_fields:
                        existing_struct.fields.append(new_field)
                        existing_seen_fields.add((new_field.name, new_field.type_name))
            else:
                substructure = Structure(substructure_name, collect_fields(field, ids_name, substructure_name, structures_list, None, excludes))
                if substructure.fields:
                    structures_list.append(substructure)
    
    return fields

def generate_structures(ids_name, xml_file, excludes=None):
    """Parse IDSDef.xml and generate C++ structures and structures.py for the specified IDS, excluding fields matching any pattern in excludes."""
    if excludes is None:
        excludes = []
    if not os.path.isfile(xml_file):
        print(f"Error: {xml_file} not found")
        return
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    header_guard = f"{ids_name.upper()}_STRUCTURES_H"
    header_file = f"{ids_name}_structures.h"
    source_file = f"{ids_name}_structures.cpp"
    structures_file = f"{ids_name}_structures.py"
    
    structures_list = []
    structure_names = set()
    
    ids_root = root.find(f".//IDS[@name='{ids_name}']")
    if ids_root is None:
        print(f"Error: {ids_name} IDS not found")
        return
    
    pre_collect_structure_names(ids_root, structure_names)
    for name in structure_names:
        if not any(s.name == name for s in structures_list):
            structures_list.append(Structure(name, []))
    
    root_fields = collect_fields(ids_root, ids_name, ids_name, structures_list, excludes=excludes)
    if root_fields:
        existing_root_struct = next((s for s in structures_list if s.name == ids_name), None)
        if existing_root_struct:
            existing_seen_fields = {(f.name, f.type_name) for f in existing_root_struct.fields}
            for field in root_fields:
                if (field.name, field.type_name) not in existing_seen_fields:
                    existing_root_struct.fields.append(field)
                    existing_seen_fields.add((field.name, field.type_name))
        else:
            structures_list.append(Structure(ids_name, root_fields))
    
    structure_dependencies = defaultdict(set)
    for struct in structures_list:
        for field in struct.fields:
            if field.is_struct:
                dep_name = field.original_name
                if dep_name != struct.name:
                    structure_dependencies[struct.name].add(dep_name)
    
    print("Debug: Structure dependencies:", {k: list(v) for k, v in structure_dependencies.items()})
    sorted_structures = topological_sort(structure_dependencies)
    
    with open(header_file, "w") as f:
        f.write(f"#ifndef {header_guard}\n#define {header_guard}\n\n")
        f.write("#include <string>\n")
        f.write("#include <vector>\n")
        f.write("#include <complex>\n\n")
        
        f.write(f"namespace {ids_name} {{\n\n")
        
        f.write(f"// Default initialization values\n")
        f.write(f"constexpr int default_value_int = {DEFAULT_INT_VALUE};\n")
        f.write(f"constexpr double default_value_double = {DEFAULT_DOUBLE_VALUE};\n")
        f.write(f'constexpr const char* default_value_string = "{DEFAULT_STRING_VALUE}";\n\n')
        
        for struct_name in sorted_structures:
            struct = next((s for s in structures_list if s.name == struct_name), None)
            if not struct:
                print(f"Warning: Structure '{struct_name}' not found in structures_list")
                continue
            f.write(f"// Structure for {struct_name}\n")
            f.write(f"// Auto-generated structure\n")
            modified_struct_name = struct_name + "_struct"
            modified_struct_name = ids_name if (struct_name == ids_name) else modified_struct_name
            f.write(f"struct {modified_struct_name} {{\n")
            
            for field in struct.fields:
                cpp_field_name = field.name.replace("-", "_")
                initializer = get_field_initializer(field.type_name, cpp_field_name, "struct_array" if field.is_vector else None)
                type_name = field.type_name + "_struct" if (not field.is_vector and field.is_struct) else field.type_name
                f.write(f"    {type_name} {cpp_field_name}{initializer};\n")
                
                if field.is_array:
                    dim = get_dimension_from_type(field.get("data_type") if hasattr(field, 'get') else "FLT_1D")
                    f.write(f"    int* {cpp_field_name}_extents = nullptr;\n")
                    f.write(f"    int {cpp_field_name}_dim = {dim};\n")
            
            f.write("};\n\n")
        
        f.write(f"}} // namespace {ids_name}\n")
        f.write(f"#endif // {header_guard}\n")
    
    print(f"Generated {header_file} successfully")
    
    with open(source_file, "w") as f:
        f.write(f'#include "{header_file}"\n')
        f.write(f"// No create functions generated; use default initialization for structures\n")
    
    print(f"Generated {source_file} successfully")
    
    with open(structures_file, "w") as f:
        f.write("# Generated structures for Python\n")
        f.write("class Field:\n")
        f.write("    def __init__(self, name: str, type_name: str, is_array: bool = False, is_vector: bool = False, is_struct: bool = False, rank: int = 0, timebasepath: str = ''):\n")
        f.write("        self.name = name\n")
        f.write("        self.type_name = type_name\n")
        f.write("        self.is_array = is_array\n")
        f.write("        self.is_vector = is_vector\n")
        f.write("        self.is_struct = is_struct\n")
        f.write("        self.extents_name = f\"{name}_extents\" if is_array else None\n")
        f.write("        self.rank = rank\n")
        f.write("        self.timebasepath = timebasepath\n\n")
        f.write("class Structure:\n")
        f.write("    def __init__(self, name: str, fields: list):\n")
        f.write("        self.name = name\n")
        f.write("        self.fields = fields\n\n")
        f.write("structures = [\n")
        
        for struct in structures_list:
            modified_struct_name = struct.name
            if struct.name != ids_name:
                modified_struct_name += "_struct" 
            f.write(f"    Structure(\"{modified_struct_name}\", [\n")
            for field in struct.fields:
                cleaned_type_name = field.type_name.replace("*", "")
                if field.is_struct and not field.is_vector:
                    cleaned_type_name += "_struct"
                escaped_timebasepath = field.timebasepath.replace('"', '\\"')
                f.write(f"        Field(\"{field.name}\", \"{cleaned_type_name}\", is_array={field.is_array}, is_vector={field.is_vector}, is_struct={field.is_struct}, rank={field.rank}, timebasepath=\"{escaped_timebasepath}\"),\n")
            f.write("    ]),\n")
        f.write("]\n")
    
    print(f"Generated {structures_file} successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate C++ structures and structures.py from IDSDef.xml for a specified IDS.")
    parser.add_argument("IDS_name", type=str, help="Name of the IDS to process (e.g., plasma_profiles)")
    parser.add_argument("xml_file", type=str, help="Path to the IDSDef.xml file")
    parser.add_argument("--excludes", type=str, nargs="*", default=[], help="List of field name patterns to exclude from the generated structures")
    args = parser.parse_args()
    generate_structures(args.IDS_name, args.xml_file, args.excludes)