import yaml

# Path to default configuration file
default_config_file_path = "../etc/default_config.yml"
# Generated configuration file path
config_file_path = "../etc/config_gen.yml"


# Class handles the configuration files
class ConfigFile:
    # Read default configuration
    @staticmethod
    def read_default_config():
        stream = open(default_config_file_path, 'r')
        file = yaml.load_all(stream)
        config_file = {}

        for doc in file:
            config_file["Min resize"] = doc["Min resize"]
            config_file["Max resize"] = doc["Max resize"]
            config_file["Min V perspective"] = doc["Min V perspective"]
            config_file["Max V perspective"] = doc["Max V perspective"]
            config_file["Min H perspective"] = doc["Min H perspective"]
            config_file["Max H perspective"] = doc["Max H perspective"]
            config_file["Min light"] = doc["Min light"]
            config_file["Max light"] = doc["Max light"]
            config_file["Min noise value"] = doc["Min noise value"]
            config_file["Max noise value"] = doc["Max noise value"]
            config_file["Min blur amplitude"] = doc["Min blur amplitude"]
            config_file["Max blur amplitude"] = doc["Max blur amplitude"]
            config_file["Vertical max aberration"] = doc["Vertical max aberration"]
            config_file["Horizontal max aberration"] = doc["Horizontal max aberration"]
            config_file["Radial distortion"] = doc["Radial distortion"]
            config_file["X center distortion"] = doc["X center distortion"]
            config_file["Y center distortion"] = doc["Y center distortion"]
            config_file["Halo amount"] = doc["Halo amount"]
            config_file["Max enlarge background vertical"] = doc["Max enlarge background vertical"]
            config_file["Max enlarge background horizontal"] = doc["Max enlarge background horizontal"]
            config_file["Otha"] = doc["Otha"]
            config_file["Black"] = doc["Black"]
            config_file["White"] = doc["White"]
            config_file["Red"] = doc["Red"]
            config_file["Use background"] = doc["Use background"]
            config_file["Use fish eye effect"] = doc["Use fish eye effect"]
            config_file["Use randomise colors"] = doc["Use randomise colors"]
            config_file["Use salt and pepper noise"] = doc["Use salt and pepper noise"]
            config_file["Crop image"] = doc["Crop image"]
            config_file["Aspect ratio"] = doc["Aspect ratio"]
            config_file["Use labels"] = doc["Use labels"]
            config_file["Labels"] = doc["Labels"]

        file.close()
        stream.close()

        return config_file

    # Write configuration file for generator
    @staticmethod
    def write_config_file(data_dict):
        file = open(config_file_path, 'w')

        # YAML configuration file label
        YAML_label = '%YAML 1.0'


        file.write(YAML_label + '\n')
        file.write("---" + "\n\n")

        file.write("Path to templates: \"{:}\"\n".format(data_dict["Path to templates"]))
        file.write("Nr of templates in folder: {:}\n\n".format(data_dict["Nr of templates in folder"]))

        if data_dict["Path to backgrounds"] == '':
            file.write("Path to backgrounds: \"{:}\"\n".format("N/A"))
            file.write("Nr of backgrounds in folder: {:}\n".format(0))
            file.write("Background names file: {:}\n\n".format("N/A"))
        else:
            file.write("Path to backgrounds: \"{:}\"\n".format(data_dict["Path to backgrounds"]))
            file.write("Nr of backgrounds in folder: {:}\n".format(data_dict["Nr of backgrounds in folder"]))
            file.write("Background names file: \"{:}\"\n\n".format(data_dict["Background names file"]))

        file.write("Output path: \"{:}\"\n\n".format(data_dict["Output path"].strip('\'')))

        file.write("Min resize: {:}\n".format(data_dict["Min resize"]))
        file.write("Max resize: {:}\n".format(data_dict["Max resize"]))
        file.write("Min H perspective: {:.1f}\n".format(data_dict["Min H perspective"]))
        file.write("Max H perspective: {:.1f}\n".format(data_dict["Max H perspective"]))
        file.write("Min V perspective: {:.1f}\n".format(data_dict["Min V perspective"]))
        file.write("Max V perspective: {:.1f}\n".format(data_dict["Max V perspective"]))
        file.write("Min light: {:}\n".format(data_dict["Min light"]))
        file.write("Max light: {:}\n".format(data_dict["Max light"]))
        file.write("Min noise value: {:}\n".format(data_dict["Min noise value"]))
        file.write("Max noise value: {:}\n".format(data_dict["Max noise value"]))
        file.write("Min blur amplitude: {:}\n".format(data_dict["Min blur amplitude"]))
        file.write("Max blur amplitude: {:}\n".format(data_dict["Max blur amplitude"]))
        file.write("Vertical max aberration: {:}\n".format(data_dict["Vertical max aberration"]))
        file.write("Horizontal max aberration: {:}\n".format(data_dict["Horizontal max aberration"]))
        file.write("Max enlarge background vertical: {:}\n".format(data_dict["Max enlarge background vertical"]))
        file.write("Max enlarge background horizontal: {:}\n".
                   format(data_dict["Max enlarge background horizontal"]))
        file.write("Radial distortion: {:.5f}\n".format(data_dict["Radial distortion"]))
        file.write("X center distortion: {:}\n".format(data_dict["X center distortion"]))
        file.write("Y center distortion: {:}\n".format(data_dict["Y center distortion"]))
        file.write("Halo amount: {:d}\n\n".format(data_dict["Halo amount"]))

        file.write("Otha: {:}\n\n".format(data_dict["Otha"]))

        file.write("Black: {:}\n".format(data_dict["Black"]))
        file.write("White: {:}\n".format(data_dict["White"]))
        file.write("Red: {:}\n\n".format(data_dict["Red"]))

        if data_dict["Path to backgrounds"] == '' or int(data_dict["Nr of backgrounds in folder"]) is 0:
            data_dict["Use background"] = False
        else:
            data_dict["Use background"] = True
        file.write("Use background: {:}\n".format(data_dict["Use background"]))
        if data_dict["Radial distortion"] == 0:
            data_dict["Use fish eye effect"] = False
        file.write("Use fish eye effect: {:}\n".format(data_dict["Use fish eye effect"]))
        file.write("Use randomise colors: {:}\n".format(data_dict["Use randomise colors"]))
        file.write("Use salt and pepper noise: {:}\n\n".format(data_dict["Use salt and pepper noise"]))

        file.write("Crop image: {:}\n".format(data_dict["Crop image"]))
        file.write("Aspect ratio: {:}\n\n".format(data_dict["Aspect ratio"]))

        file.write("Signs: {:}\n".format(data_dict["Signs"]))
        file.write("Use labels: {:}\n".format(data_dict["Use labels"]))
        file.write("Labels: {:}\n".format(data_dict["Labels"]))

        file.close()
