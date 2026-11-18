import xml.etree.ElementTree as ET
import argparse

def find_element_by_path(root, path, namespaces=None):
    """Find an element in the XML tree by its path or name attribute, supporting namespaces."""
    print(f"Searching for path '{path}' in element {root.tag}")
    current = root
    for part in path.split('/'):
        found = None
        print(f"  Processing path part '{part}'")
        # Try non-namespaced tag first
        print(f"    Trying tag '{part}' (no namespace)")
        found = current.find(part)
        
        # If not found, try namespaced tag
        if found is None and namespaces:
            for ns in namespaces.values():
                tag = f'{{{ns}}}{part}'
                print(f"    Trying tag '{tag}'")
                found = current.find(tag)
                if found is not None:
                    print(f"    Found element with tag '{tag}'")
                    break
        
        # If not found by tag, try finding a 'field' element with matching name attribute
        if found is None:
            print(f"    Tag '{part}' not found, checking for field with name='{part}'")
            # Try non-namespaced field first
            print(f"    Checking fields with tag 'field' (no namespace)")
            for field in current.findall('field'):
                field_name = field.get('name')
                print(f"      Found field with name='{field_name}'")
                if field_name == part:
                    found = field
                    print(f"      Matched field with name='{part}'")
                    break
            # Then try namespaced fields
            if found is None and namespaces:
                for ns in namespaces.values():
                    field_tag = f'{{{ns}}}field'
                    print(f"    Checking fields with tag '{field_tag}'")
                    for field in current.findall(field_tag):
                        field_name = field.get('name')
                        print(f"      Found field with name='{field_name}'")
                        if field_name == part:
                            found = field
                            print(f"      Matched field with name='{part}'")
                            break
                    if found is not None:
                        break
        
        if found is None:
            print(f"    Failed to find element or field with name='{part}'")
            return None
        current = found
    print(f"Found element for path '{path}': {current.tag}")
    return current

def extract_ids(input_file, output_file, ids_name, fields=None):
    # Parse the input XML file
    try:
        tree = ET.parse(input_file)
    except ET.ParseError as e:
        print(f"Error parsing XML file {input_file}: {e}")
        return
    root = tree.getroot()

    # Namespace handling
    namespaces = {'fn': 'http://www.w3.org/2005/02/xpath-functions'}
    # Discover all namespaces in the XML, including default namespace
    default_ns = None
    if '}' in root.tag:
        default_ns = root.tag.split('}')[0][1:]
        namespaces['default'] = default_ns
    for elem in root.iter():
        if '}' in elem.tag:
            ns = elem.tag.split('}')[0][1:]
            if ns and ns not in namespaces.values():
                namespaces[f'ns{len(namespaces)}'] = ns
    print(f"Detected namespaces: {namespaces}")

    # Create a new root element with the same tag and attributes
    new_root = ET.Element(root.tag, attrib=root.attrib)

    # Find and copy the specified IDS element
    ids_found = False
    # Try non-namespaced IDS first
    for ids_elem in root.findall('IDS'):
        if ids_elem.get('name') == ids_name:
            ids_found = True
            print(f"Found IDS '{ids_name}' with attributes: {ids_elem.attrib}")
            print("Child elements of IDS:")
            for child in ids_elem:
                print(f"  {child.tag} (name: {child.get('name', 'N/A')})")
            
            if fields:
                # Create a new IDS element
                new_ids = ET.Element('IDS', attrib=ids_elem.attrib)
                # Copy only specified fields
                for field in fields:
                    field_elem = find_element_by_path(ids_elem, field, namespaces)
                    if field_elem is not None:
                        print(f"Found field '{field}' with attributes: {field_elem.attrib}")
                        # Create parent structure if needed
                        current = new_ids
                        path_parts = field.split('/')
                        for i, part in enumerate(path_parts[:-1]):
                            parent_path = '/'.join(path_parts[:i + 1])
                            parent = find_element_by_path(new_ids, parent_path, namespaces)
                            if parent is None:
                                # Create parent element
                                orig_parent = find_element_by_path(ids_elem, parent_path, namespaces)
                                if orig_parent is not None:
                                    parent = ET.SubElement(current, orig_parent.tag, attrib=orig_parent.attrib)
                                else:
                                    parent = ET.SubElement(current, 'field', attrib={'name': part})
                            current = parent
                        # Add the field element
                        current.append(field_elem)
                    else:
                        print(f"Warning: Field '{field}' not found in IDS '{ids_name}'")
                new_root.append(new_ids)
            else:
                # Copy the entire IDS element if no fields specified
                new_root.append(ids_elem)
    
    # Try namespaced IDS if not found
    if not ids_found and namespaces:
        for ns in namespaces.values():
            for ids_elem in root.findall(f'{{{ns}}}IDS'):
                if ids_elem.get('name') == ids_name:
                    ids_found = True
                    print(f"Found IDS '{ids_name}' with attributes: {ids_elem.attrib}")
                    print("Child elements of IDS:")
                    for child in ids_elem:
                        print(f"  {child.tag} (name: {child.get('name', 'N/A')})")
                    
                    if fields:
                        new_ids = ET.Element('IDS', attrib=ids_elem.attrib)
                        for field in fields:  # Fixed typo: 'KE fields' -> 'fields'
                            field_elem = find_element_by_path(ids_elem, field, namespaces)
                            if field_elem is not None:
                                print(f"Found field '{field}' with attributes: {field_elem.attrib}")
                                current = new_ids
                                path_parts = field.split('/')
                                for i, part in enumerate(path_parts[:-1]):
                                    parent_path = '/'.join(path_parts[:i + 1])
                                    parent = find_element_by_path(new_ids, parent_path, namespaces)
                                    if parent is None:
                                        orig_parent = find_element_by_path(ids_elem, parent_path, namespaces)
                                        if orig_parent is not None:
                                            parent = ET.SubElement(current, orig_parent.tag, attrib=orig_parent.attrib)
                                        else:
                                            parent = ET.SubElement(current, 'field', attrib={'name': part})
                                    current = parent
                                current.append(field_elem)
                            else:
                                print(f"Warning: Field '{field}' not found in IDS '{ids_name}'")
                        new_root.append(new_ids)
                    else:
                        new_root.append(ids_elem)
                    break
            if ids_found:
                break

    if not ids_found:
        print(f"Error: IDS '{ids_name}' not found in the XML file")

    # Create a new tree with the modified root
    new_tree = ET.ElementTree(new_root)

    # Write the new XML file with proper XML declaration and encoding
    with open(output_file, 'wb') as f:
        new_tree.write(f, encoding='UTF-8', xml_declaration=True)

def main():
    parser = argparse.ArgumentParser(description='Extract specific IDS and fields from XML file')
    parser.add_argument('input_file', help='Input XML file path')
    parser.add_argument('output_file', help='Output XML file path')
    parser.add_argument('ids_name', help='Name of the IDS to extract')
    parser.add_argument('--fields', nargs='*', help='Specific fields to extract from the IDS (use / for nested paths)')
    
    args = parser.parse_args()
    
    extract_ids(args.input_file, args.output_file, args.ids_name, args.fields)

if __name__ == '__main__':
    main()