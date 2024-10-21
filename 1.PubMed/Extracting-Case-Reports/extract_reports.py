import re
import urllib.request
import os
import xml.etree.ElementTree as ET
import requests
import tarfile
from collections import defaultdict
import matplotlib.pyplot as plt
import shutil
from bs4 import BeautifulSoup
import numpy as np
import json
from tqdm import trange, tqdm
import nltk

def download_tarballs_from_pmcids(file_path, download_directory):

    """
    Main download function : Download the files required.

    Args:
    file_path (str) : Path to the file that contains the PMCIDs.
    download_directory (str) : Name of the main directory where we want to download the data.
    """

    create_download_directory(download_directory)
    pmc_ids = extract_pmcids_from_file(file_path)
    for pmc_id in pmc_ids:
        tarball_link = get_tarball_link(pmc_id)
        if tarball_link:
            filename = f"{pmc_id}.tar.gz"
            download_tarball(tarball_link, filename, download_directory)
            print(f"Downloaded : {filename}")
        else:
            print(f"Download of the file impossible for PMCID : {pmc_id}")


def download_tarball(tarball_link, filename, download_directory):

    """
    Child function of download_tarballs_from_pmcids : Download one file.

    Args:
    tarball_link (str) : Link that is being use to download the file.
    filename (str) : Name of the file/sub-directory we want to create.
    download_directory (str) : Name of the main directory where we want to download the file/sub-directory.
    """

    urllib.request.urlretrieve(tarball_link, os.path.join(download_directory, filename))

def get_tarball_link(pmcid):

    """
    Child function of download_tarball : Get the links that will be used to download the files from PubMed FTP OA, using the PMCIDs 
    and urls related.

    Args:
    pmcid (str) : Identifier of the article we want to download.
    """

    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={}"

    try:
        response = requests.get(url.format(pmcid))
        xml_content = response.content
        root = ET.fromstring(xml_content)
        link_element = root.find(".//link")
        if link_element is not None:
            href_attribute = link_element.get("href")
            return href_attribute
        else:
            print(f"No link found for {pmcid}")
            return None
    except Exception as e:
        print(f"An error occured while getting the link for {pmcid}: {e}")
        return None
    
def create_download_directory(directory):

    """
    Child function of download_tarballs_from_pmcids : Create a folder that will contain the .tar.gz files we extract.

    Args:
    directory (str) : Path to the folder that will contain the .tar.gz files.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_pmcids_from_file(file_path):

    """
    Child function of download_tarballs_from_pmcids : Extract PMCIDs from any file containing PMCIDs.

    Args:
    file_path (str) : Path to the file that contains the PMCIDs.
    """

    with open(file_path, 'r') as file:
        text = file.read()
    pmc_pattern = re.compile(r'\bPMC\d+\b')
    return pmc_pattern.findall(text)


def extract_xml_from_tar(download_directory, xml_folder):
    
    """
    Main extraction function : Extracts XML files from tar.gz archives in the specified folder.

    Args:
    download_directory (str) : Path to the folder containing the tar.gz files.
    xml_folder (str) : Path to the folder where XML files will be extracted.
    """

    create_download_directory(xml_folder)

    # Iterate through all tar.gz files in the tar_folder
    for filename in os.listdir(download_directory):
        tar_filepath = os.path.join(download_directory, filename)
        
        # Extract XML files
        with tarfile.open(tar_filepath, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith('.nxml'):
                    # Extract the XML file to the xml_folder
                    tar.extract(member, path=xml_folder)

def create_groups_and_plot(titles_list, xml_folder):

    """
    Main function to create a bar plot containing groups of titles (groups because sometimes spelling is different but it has the same meaning.)
    i.e : "Patient et Observation" and "Patients et Observations" will be put in the same group. To do so we use a child function called jaccard_similarity.

    Args:
    titles_list (str) : List of titles we gathered through traverse_folder.
    xml_folder (str) : The folder on which we wanna have the plot.
    """

    groupes_titres = defaultdict(list)

    seuil_similarity = 0.8

    for titre in titles_list:
        if titre == "Patient and observation" or titre == "Patient and Observation":
            continue
        else:
            groupe_trouve = False
            for groupe, membres in groupes_titres.items():
                for membre in membres:
                    similarity = jaccard_similarity(membre, titre)
                    if similarity >= seuil_similarity:
                        groupes_titres[groupe].append(titre)
                        groupe_trouve = True
                        break
                if groupe_trouve:
                    break
            if not groupe_trouve:
                groupes_titres[titre].append(titre)

    group_occurrences = {groupe: len(membres) for groupe, membres in groupes_titres.items()}

    group_occurrences = {groupe: occ for groupe, occ in group_occurrences.items() if occ is not None}

    groupes = list(group_occurrences.keys())
    occurrences = list(group_occurrences.values())

    for groupe, taille in group_occurrences.items():
        print(f"Group: {groupe}")
        print(f"Size of the group: {taille}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(groupes)), occurrences, tick_label=groupes, color='skyblue')
    plt.xlabel('Groups')
    plt.ylabel('Occurrences')
    plt.title('Occurrences of the groups')
    plt.xticks(rotation=90)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'plot_{xml_folder}.png')

    return groupes_titres

def traverse_folder(xml_folder):

    """
    Main function to check the titles after the first <sec> section in every file. Print the title found for each file. Print aswell
    how many files correspond to what we desire and how many files doesn't.

    Child function of create_groups_and_plot, which returns the arguent titles_list used in create_groups_and_plot.

    Save all the titles aswell.

    Args:
    XML_file_path (str) : Files within an XML folder with the .nxml format.
    """

    titles_list =[]
    k=0
    count = 0
    count_2 = 0
    for root, dirs, files in os.walk(xml_folder):
        for file in files:
            print(k)
            print(os.path.join(root, file))
            if file.endswith('.nxml'):
                k+=1
                file_path = os.path.join(root, file)
                next_section_title = extract_section_after_intro(file_path)
                titles_list.append(next_section_title)
                if next_section_title:
                    print(f"Title of next section in {os.path.join(root, file)} : {next_section_title}")
                    count += 1
                else:
                    print(f"No section was found after the introduction in {os.path.join(root, file)}.")
                    count_2 += 1

    print(f"We found {count} files corresponding to what we expected.")
    print(f"There is an error {count_2} files.")
    return titles_list


def extract_section_after_intro(xml_file_path):

    """
    Child function of traverse_folder allowing us to extract the title of the section coming after the introduction <sec> section.

    Args:
    XML_file_path (str) : Files within an XML folder with the .nxml format.
    """

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    body_tag = root.find(".//body")
    if body_tag is None:
        return None

    intro_tag = None
    for sec in body_tag.findall(".//sec"):
        title_tag = sec.find("title")
        if title_tag is not None and title_tag.text is not None and title_tag.text.strip().lower() in ['introduction', 'intro']:
            intro_tag = sec
            break

    if intro_tag is None:
        return None

    next_section_tag = None
    intro_tag_found = False
    for sec_tag in body_tag.findall(".//sec"):
        if intro_tag_found and sec_tag.get('id') != intro_tag.get('id'):
            next_section_tag = sec_tag
            break
        if sec_tag.get('id') == intro_tag.get('id'):
            intro_tag_found = True

    if next_section_tag is None:
        return None

    next_section_title_tag = next_section_tag.find("title")
    if next_section_title_tag is not None:
        return next_section_title_tag.text.strip()

    return None

def jaccard_similarity(s1, s2):

    """
    Child function of creer_groupes_et_plot to check the similarity between titles.
    Takes characters in a set of words and comapre it to another set of words.
    We get a similarity score that is better the more it approach 1.
    The jaccard similarity is being calculated as follows : j_s = |A inter B| / |A union B|

    Args:
    s1, s2 (str) : Set of words used to compare the similarity score.
    """

    if s1 is None or s2 is None:
        return 0
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def XML_size_analysis_and_visualisation(xml_folder):

    """
    Print statistics from our files in the desired directory : Mean,min,max of size, along with the name of the files concerned.
    Plot in a seperate .png file the boxplot of the size of our files contained in the directory.

    Args:
    xml_folder (str) : Path to the XML folder containing the desired files.
    """

    file_sizes = []
    file_links = []

    for subdir in os.listdir(xml_folder):
        subdir_path = os.path.join(xml_folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".nxml"):
                    file_path = os.path.join(subdir_path, filename)
                    file_stats = os.stat(file_path)
                    file_info = (subdir_path, filename, file_stats.st_size)
                    file_links.append(file_info)
                    file_sizes.append((filename, file_stats.st_size))

    total_size = sum(size for _, size in file_sizes)
    average_size = total_size / len(file_sizes)
    min_file_info = min(file_links, key=lambda x: x[2])
    max_file_info = max(file_links, key=lambda x: x[2])

    print(f"Average size of files : {average_size} octets")
    print(f"Name of the sub-folder containing the minimum sized file : {os.path.basename(min_file_info[0])}")
    print(f"Name of the minimum sized file : {min_file_info[1]}")
    print(f"Minimum size of file : {min_file_info[2]}")
    print(f"Name of the sub-folder containing the maximum sized file : {os.path.basename(max_file_info[0])}")
    print(f"Name of the maximum sized file  : {max_file_info[1]}")
    print(f"Maximum size of file  : {max_file_info[2]}")

    file_sizes_plot = [file_info[2] for file_info in file_links]
    plt.boxplot(file_sizes_plot)
    plt.title('Boxplot of File Sizes')
    plt.ylabel('File Size (bytes)')
    plt.savefig(f"Boxplot_{os.path.basename(xml_folder)}.png")

    Q1 = np.percentile(file_sizes_plot, 25)
    Q3 = np.percentile(file_sizes_plot, 75)

    print("Q1:", Q1)
    print("Q3:", Q3)

    return file_links, file_sizes, average_size, min_file_info, max_file_info, file_sizes_plot, Q1, Q3

def restricting_files(Q1, Q3, desired_size, file_links, output_file_1, output_file_2, output_file_3):

    """
    Child function of XML_size_analysis_and_visualisation later allowing us to create a folder without unwanted files.

    Args:
    Q1 (int) : First quartile of the size of our files.
    Q3 (int) : Third quartile of the size of our files.
    desired_size (int) : Minimum size we want our files to be.
    file_links (list) : Contains (subdir_path, filename, file_stats.st_size).
    output_file_1 (str) :  Name of the file containing (subdir_path, filename, file_stats.st_size) of the files under Q1.
    output_file_2 (str) :  Name of the file containing (subdir_path, filename, file_stats.st_size) of the files bewteen Q1 and desired shape.
    output_file_3 (str) :  Name of the file containing (subdir_path, filename, file_stats.st_size) of the files over Q3.
    """

    files_under_Q1 = []
    files_in_between_Q1_and_desired_shape = []
    files_above_Q3 = []

    for file_info in file_links:
        if file_info[2] <= Q1:
            files_under_Q1.append(file_info)
        elif Q1 < file_info[2] <= desired_size:
            files_in_between_Q1_and_desired_shape.append(file_info)
        elif file_info[2] > Q3:
            files_above_Q3.append(file_info)

    files_under_Q1 = sorted(files_under_Q1, key=lambda x: x[2], reverse=True)
    files_in_between_Q1_and_desired_shape = sorted(files_in_between_Q1_and_desired_shape, key=lambda x: x[2], reverse=True)
    files_above_Q3 = sorted(files_above_Q3, key=lambda x: x[2], reverse=True)

    with open(output_file_1, 'w', encoding='utf-8') as file:
        for item in files_under_Q1:
            file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

    with open(output_file_2, 'w', encoding='utf-8') as file:
        for item in files_in_between_Q1_and_desired_shape:
            file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

    with open(output_file_3, 'w', encoding='utf-8') as file:
        for item in files_above_Q3:
            file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

    return output_file_1, output_file_2, output_file_3

def XML_separate(output_file_1, output_file_2, output_file_3):

    """
    Main function to separate our files based on their sizes. ie if the size of a file is under the first Quartile, it will put it in a new folder.

    Args:
    output_file_1 (str) : Files .txt containing the info of where to find the files under the 1st Quartile in the main XML_folder.
    output_file_2 (str) : Files .txt containing the info of where to find the files between the 1st Quartile and some desired size in the main XML_folder.
    output_file_3 (str) : Files .txt containing the info of where to find the files above the 3rd Quartile in the main XML_folder.
    """

    under_Q1_folder = input("Enter a name for the folder containing files of size < Q1 : ")
    between_Q1_and_desired_folder = input("Enter a name for the folder of size between Q1 and desired size : ")
    above_Q3_folder = input("Enter a name for the folder of size > Q3 : ")

    under_Q1_path = os.path.join(os.getcwd(), under_Q1_folder)
    between_Q1_and_desired_path = os.path.join(os.getcwd(), between_Q1_and_desired_folder)
    above_Q3_path = os.path.join(os.getcwd(), above_Q3_folder)
    
    if not os.path.exists(under_Q1_path):
        os.makedirs(under_Q1_path)

    if not os.path.exists(between_Q1_and_desired_path):
        os.makedirs(between_Q1_and_desired_path)

    if not os.path.exists(above_Q3_path):
        os.makedirs(above_Q3_path)
    
    # Copy files to appropriate folders
    for output_file, folder in zip([output_file_1, output_file_2, output_file_3], [under_Q1_path, between_Q1_and_desired_path, above_Q3_path]):
        with open(output_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(', ')
                subdir_path = parts[0]
                filename = parts[1]
                # Extract PMC number from subdir_path
                match = re.search(r'PMC(\d+)', subdir_path)
                if match:
                    pmc_number = "PMC" + match.group(1)
                    if filename.endswith('.nxml'):
                        src_path = os.path.join(subdir_path, filename)
                        dst_path = os.path.join(folder, pmc_number)
                        if not os.path.exists(dst_path):
                            os.makedirs(dst_path)
                        shutil.copy(src_path, dst_path)

    return under_Q1_folder, between_Q1_and_desired_folder, above_Q3_folder


def get_length_directory(directory):

    """
    Main function to get the number of files inside a folder.

    Args:
    directory (str) : Directory containing the files .nxml we want to look at.
    """

    if os.path.exists(directory):
        max_size = 0
        nb_sous_dossiers = 0
        for root, dirs, files in os.walk(directory):
            nb_sous_dossiers += len(dirs)
            for file in files:
                size = os.path.getsize(os.path.join(root, file))
                if size > max_size:
                    max_size = size

        print(f"Maximal size of a file in the directory {directory} is {max_size} octets.")
        print(f"It contains{nb_sous_dossiers} sub-directories.")

def extract_title(xml_content):

    """
    Child function of search_for_unwanted_title allowing us to extract the first title of the <sec> tags.

    Args:
    XML (str) : XML content inside of XML folder.
    """

    root = ET.fromstring(xml_content)
    title_element = root.find(".//sec//title")
    if title_element is not None and title_element.text is not None:
        return title_element.text.strip()
    else:
        return "No title found"

def search_for_unwanted_title(directory,title_pattern,titre):

    """
    Main function to search for unwanted title that will print each title for every file.
    Will print how many titles corresponds to the unwanted title we are looking for.

    Args:
    directory (str) : XML folder containing our files.
    title_pattern (str) : re.compile of the possible ways our title can be write.
    titre (str) : Unwanted title we are looking for.
    """

    counting = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.nxml'):
                    file_path = os.path.join(subdir_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        title = extract_title(content)
                        if title is None:
                            continue
                        print(f"Subdir : {subdir},File: {filename}, Title: {title}")
                        if title_pattern.search(title):
                            counting += 1

    print(f"From all the files, {counting} have for title {titre}.")

def tag_occurrence(directory):

    """
    Main function to get the number of occurences for the total directory we are dealing with.
    Print how many tags there is in the folder, aswell as the number of different tags and how many times they occur.

    Args:
    directory (str) : XML folder containing our files.
    """

    tag_counts = {}

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.nxml'):
                    file_path = os.path.join(subdir_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        soup = BeautifulSoup(content, 'xml')
                        tags = soup.find_all()
                        for tag in tags:
                            tag_name = tag.name
                            if tag_name in tag_counts:
                                tag_counts[tag_name] += 1
                            else:
                                tag_counts[tag_name] = 1

    total_tags = sum(tag_counts.values())
    print("Total number of tags :", total_tags)
    print("Number of different tags :", len(tag_counts))
    print("Statistics on tags occurence :")
    sorted_tag_counts = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tag_counts:
        print(f"{tag}: {count} occurrences")

def concat_xml_folders(input_folders, output_folder):

    """
    Main function to concatenate XML folders together.

    Args:
    input_folders (list of str) : XML folders containing the files we want to concatenate.
    output_folder (str) : XML folder containing files resulting from the concatenation.
    """

    for input_folder in input_folders:
        for subdir, _, files in os.walk(input_folder):
            relative_path = os.path.relpath(subdir, input_folder)
            output_subdir = os.path.join(output_folder, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            for filename in files:
                if filename.endswith('.nxml'):
                    tree = ET.parse(os.path.join(subdir, filename))
                    root = tree.getroot()
                    combined_root = root
                    for other_input_folder in input_folders:
                        if other_input_folder == input_folder:
                            continue
                        corresponding_file = os.path.join(other_input_folder, relative_path, filename)
                        if os.path.exists(corresponding_file):
                            choice = input("The directories {} and {} exists. Choose which one to keep (1 for the first, 2 for the second): ".format(filename, os.path.basename(corresponding_file)))
                            if choice == '1':
                                continue
                            elif choice == '2':
                                other_tree = ET.parse(corresponding_file)
                                other_root = other_tree.getroot()
                                combined_root.extend(other_root)

                    combined_tree = ET.ElementTree(combined_root)
                    combined_tree.write(os.path.join(output_subdir, filename))


def final_xml(xml_folder, output_folder, desired_size, title_pattern, unwanted_title):

    """
    Main function that will run the complete process from XML analysis to final XML concatenation.
    
    Args:
    xml_folder (str): Path to the main folder containing XML files.
    output_folder (str): Path to the folder where final XML files will be saved.
    desired_size (int): Minimum size for filtering files.
    title_pattern (str): Regex pattern to detect unwanted titles.
    unwanted_title (str): Specific title string to search for.
    """
    
    # 1. Analyse the XML sizes and get stats
    print("Step 1: XML Size Analysis and Visualization")
    file_links, file_sizes, avg_size, min_file_info, max_file_info, file_sizes_plot, Q1, Q3 = XML_size_analysis_and_visualisation(xml_folder)

    # 2. Filter files by size and save the file information in three categories
    print("Step 2: Restricting Files Based on Size")
    output_file_1 = 'files_under_Q1.txt'
    output_file_2 = 'files_between_Q1_and_desired_size.txt'
    output_file_3 = 'files_above_Q3.txt'
    restricting_files(Q1, Q3, desired_size, file_links, output_file_1, output_file_2, output_file_3)

    # 3. Separate files into different folders based on their size categories
    print("Step 3: Separating Files into Folders Based on Size")
    XML_separate(output_file_1, output_file_2, output_file_3)

    # 4. Search for unwanted titles in the XML files
    print("Step 4: Searching for Unwanted Titles")
    title_pattern_compiled = re.compile(title_pattern)
    search_for_unwanted_title(xml_folder, title_pattern_compiled, unwanted_title)

    # 5. Analyze tag occurrences in the XML files
    print("Step 5: Tag Occurrence Analysis")
    tag_occurrence(xml_folder)

    # 6. Concatenate XML files from all subfolders into a final output folder
    print("Step 6: Concatenating XML Files into Final Folder")
    input_folders = [xml_folder, output_file_1, output_file_2, output_file_3]
    concat_xml_folders(input_folders, output_folder)

    print("Process completed. Final XML files are saved in:", output_folder)

nltk.download('punkt')

p_nodes = ['p', 'list-item', 'disp-quote', "AbstractText"]
sec_nodes = ['sec', 'list']
inline_elements = ['italic', 'bold', 'sup', 'strike', 'sub', 'sc', 'named-content', 'underline', \
    'statement', 'monospace', 'roman', 'overline', 'styled-content']
# Detect label in titles, such as "3.1" in "3.1 Patient 1"
label_pattern = re.compile(r'^[0-9]\.?[0-9]?\.?[0-9]?\.? ?')
# Replace several kinds of whitespace character into ' ' for convieniece of further processing. 
space = r"[\u3000\u2009\u2002\u2003\u00a0\u200a\xa0]"


def clean_text(text):

    """
    Deal with unexpected and rebundant whitespace.
    Input:
        text: text to be cleaned.
    Output:
        cleaned text.
    """

    text = re.sub(space, ' ', text).replace(u'\u2010', '-').strip()
    text = re.sub(r" +", ' ', text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\(, \)+", "", text)
    text = re.sub(r"\(,\s*\){2,}", "", text)
    text = re.sub(r"\(\s*,\s*\)", "", text)
    text = re.sub(r"\[\s*,\s*\]", "", text)
    text = re.sub(r"\(\s*et\s*\)", "", text)
    text = re.sub(r"\[\s*et\s*\]", "", text)
    text = re.sub(r"\(\s*and\s*\)", "", text)
    text = re.sub(r"\[\s*and\s*\]", "", text)
    text = re.sub(r"\(,\s*and\s*\)", "", text)
    text = re.sub(r"\(,\s*et\s*\)", "", text)
    text = re.sub(r"\((\s*,\s*){2,}\)", "", text)
    text = re.sub(r"\(\s*,\s*(,\s*)+\)", "(", text)
    text = re.sub(r"\[-]", "", text)
    text = re.sub(r"\(A\)", "", text)
    text = re.sub(r"\( A\)", "", text)
    text = re.sub(r"\(A \)", "", text)
    text = re.sub(r"\(A, B\)", "", text)
    text = re.sub(r"\( A, B\)", "", text)
    text = re.sub(r"\( A,B\)", "", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = text.replace('[, , , ]', "")
    text = text.replace('[, , ]', "")
    text = text.replace('[,]', "")
    text = text.replace('(,', "")
    text = text.replace(',)', "")
    return text


def is_english(text):
    # Check for common English sentence structures
    pattern = r'\b(man|woman|we|the)\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

def getTitle(sec):

    """
    Extract title in a title node.
    Input:
        sec: An element node of type 'sec'.
    Output:
        title of the section or string "" if no title node detected.
    """

    for child in sec:
        if child.tag == "title":
            title = getText(child)
            return clean_text(re.sub(label_pattern, '', title))
    return ""


def getText(para):

    """
    Extract text from a given node and its successive children.
    Input:
        para: An element node of type 'p' or others.
    Output:
        text within the node and its successive children.
    """ 

    text = para.text if para.text else ""
    for child in para:
        if child.tag in inline_elements:
            text += child.text if child.text else ""
        if child.tag in sec_nodes or child.tag in p_nodes:
            text += getText(child) + ' '
        text += child.tail if child.tail else ""

    cleaned_text = clean_text(text)

    if is_english(cleaned_text):
        return ""
    
    return clean_text(text)


def parse_paragraph(body, secname=""):
    """
    Parse paragraph for an article or section. 
    Input:
        body: Element node to be parsed. If an article is to be parsed, input body node of xml.
        secname: Section name that will be concatenated ahead to all parsed paragraph.
            If an article is to be parsed, input empty string.
    Output:
        List of tuples (titles, text), where titles are section names (if subsection involved, titles are seperated by '[SEP]' token).
    """

    results = []
    title = getTitle(body)
    titles = secname + title
    if title:
        titles += "[SEP]"

    for child in body:
        if child.tag in sec_nodes:
            break
        if child.tag in p_nodes:
            text = getText(child)
            if len(text) > 1:
                results.append((titles, text))
        elif child.tag in sec_nodes:
            results += parse_paragraph(child, titles)

    return results

def getSection(sec):

    """
    Extract text of a section.
    Input:
        sec: Element node of type 'sec'.
    Output:
        Texts within this section, paragraphs seperated by '\n'.
    """

    paras = parse_paragraph(sec)
    text = ""
    for para in paras:
        text += para[1] + '\n'
    return clean_text(text)

def getLongestSentence(sec):

    """
    Extract the length of a section.
    Input:
        sec: Element node of type 'sec'.
    Output:
        Length of the said section.
    """

    text = getSection(sec)
    sentences = nltk.sent_tokenize(text)
    if sentences:
        longest_sentence = max(sentences, key=len)
        longest_sentence_length = len(longest_sentence)
        return longest_sentence_length
    else:
        return None


def extract_text_from_xml_folder(folder_path):

    """
    Extract text from an XML folder and get the output in a .json file. Each set will contain the PMCID, and the first two 'sec' sections (Introduction with
    title and text within the section and Patient and Observation with title and text within the section).
    Args:
        folder_path : Path to the folder we want to extract the text from.
    Output:
        Texts about the first two sections (introduction and Patient and observation) within all the XML folder.
    """

    c = 0
    extracted_text = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".nxml"):
                file_path = os.path.join(root, file_name)
                pmc_number = root.split("/")[-1]
                with open(file_path, "r", encoding="utf-8") as file:
                    xml_content = file.read()
                root_element = ET.fromstring(xml_content)
                body = root_element.find(".//body")
                if body is not None:
                    sections = body.findall(".//sec")
                    intro_section = {}
                    next_section = None
                    found_intro = False
                    for sec in sections:
                        title = getTitle(sec)
                        if title.lower() == "introduction" or title.lower() == "intro":
                            c += 1
                            intro_section["title"] = title
                            intro_section["text"] = getSection(sec)
                            intro_section["longest_sequence"] = getLongestSentence(sec)
                            found_intro = True
                        elif found_intro:
                            if title.lower() == "discussion" or title.lower() == "conclusion":
                                break
                            if next_section is None:
                                next_section = {"title": title, "text": getSection(sec), "subsections" : []}
                            else:
                                next_section["subsections"].append({"title" : getTitle(sec), "text" : getSection(sec)})
                    if next_section is not None:
                        extracted_text.append({
                            "root": pmc_number,
                            "introduction": intro_section,
                            "next_section": next_section
                        })
    print(f"We extracted in a .json file {c} files.")
    return extracted_text

file_path = input("Please enter the path of the search file containing PMCIDs (e.g., pubmed_search.txt): ")
download_directory = input("Please enter the name of the directory where you want to store those PMC (e.g., PMC): ")
xml_folder = input("Please enter the name of the XML folder where the .nxml files will be located (e.g., XML): ")

# Calling our functions :
download_tarballs_from_pmcids(file_path, download_directory)
extract_xml_from_tar(download_directory, xml_folder)

# Ask the user to input the parameters
output_folder = input("Please enter the name of the final folder (e.g., XML_Final): ")
while True:
    try:
        desired_size = int(input("Please enter the maximum size of the files you want (e.g., 50000): "))
        break  # Exit loop if conversion is successful
    except ValueError:
        print("Invalid input. Please enter an integer value.")
title_pattern = input("Please enter the title pattern (e.g., Discussion): ")
unwanted_title = input("Please enter the unwanted title (e.g., Discussion): ")


# Launch the final procedure
final_xml(xml_folder, output_folder, desired_size, title_pattern, unwanted_title)

# Extract the text from XML files
extracted_data = extract_text_from_xml_folder(output_folder)

# Write the extracted data to a JSON file
output_file = "extracted_data_structure.json"
with open(output_file, "w") as json_file:
    json.dump(extracted_data, json_file, indent=4)

print("Extraction complete. Data saved to:", output_file)

with open(output_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Définir une fonction pour décoder les caractères échappés
def decode_escaped_characters(text):
    return text.encode('utf_8').decode('utf-8')

# Appliquer la fonction aux sections de texte
for item in data:
    if 'introduction' in item:
        item['introduction']['text'] = decode_escaped_characters(item['introduction']['text'])
    if 'next_section' in item:
        item['next_section']['text'] = decode_escaped_characters(item['next_section']['text'])
        for subsection in item['next_section'].get('subsections', []):
            subsection['text'] = decode_escaped_characters(subsection['text'])

# Enregistrer les données décodées
with open('cleaned_data_structure.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

input_file = 'cleaned_data_structure.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Remove entries with empty text in both 'introduction' and 'next_section'
filtered_data = [
    entry for entry in data if not (
        entry['introduction']['text'] == "" and 
        entry['next_section']['text'] == ""
    )
]
# Overwrite the input file with the filtered data
with open(input_file, 'w', encoding='utf-8') as json_file:
    json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)
