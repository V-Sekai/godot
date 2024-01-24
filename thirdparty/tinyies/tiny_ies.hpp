/*
MIT License

Copyright(c) 2021 - 2022 Lukas Lipp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <limits>
#include <algorithm>

template <typename T>
class tiny_ies {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "T must be float or double");
public:
    struct light {
        light() : lamp_to_luminaire_geometry(),
            number_of_tilt_angles(0),
            number_lights(0),
            lumens_per_lamp(0),
            multiplier(0),
            number_vertical_angles(0),
            number_horizontal_angles(0),
            photometric_type(0), units_type(0),
            width(0), length(0), height(0),
            ballast_factor(0),
            future_use(0),
            input_watts(0),
            min_vertical_angle(0),
            max_vertical_angle(0),
            min_horizontal_angle(0),
            max_horizontal_angle(0),
            max_candela(0)
        {
        }

        /*
        ** HEADER
        */
        std::string ies_version;
        std::map<std::string, std::string> properties;
        std::string tilt;

        /*
        ** DATA
        */
        /* tilt data */
        int lamp_to_luminaire_geometry;
        int number_of_tilt_angles;
        std::vector<T> tilt_angles;
        std::vector<T> tilt_multiplying_factors;

        /* data */
        int number_lights;
        int lumens_per_lamp;
        T multiplier;
        int number_vertical_angles;
        int number_horizontal_angles;
        int photometric_type;
        int units_type;
        T width;
        T length;
        T height;

        T ballast_factor;
        T future_use;
        T input_watts;

        std::vector<T> vertical_angles;
        T min_vertical_angle;
        T max_vertical_angle;
        std::vector<T> horizontal_angles;
        T min_horizontal_angle;
        T max_horizontal_angle;

        std::vector<T> candela;
        T max_candela;
    };

    static bool load_ies_from_buffer(const std::string& bufferInput, std::string& err_out, std::string& warn_out, light& ies_out) {
       std::istringstream f(bufferInput);
        if (!f) {
            err_out = "Failed reading file.";
            return false;
        }

        ies_out = light();
        std::string line;
        /*
        ** HEADER
        */
        // the first line in a valid ies file should be IESNA:
        if (f.good()) {
            std::getline(f, line);
            if (!read_property("IESNA", line, ies_out.ies_version)) {
                warn_out = "First line did not start with IESNA.";
            }
        }
        // read properties
        while (f.good() && std::getline(f, line)) {
            if (read_property("TILT=", line, ies_out.tilt)) break;
            read_property(line, ies_out.properties);
        }
        if (ies_out.tilt.empty()) {
            err_out = "TILT property not found.";
            return false;
        }

        /*
        ** DATA
        */
        // replace comma with empty space
        std::ostringstream oss;
        oss << f.rdbuf();
        std::string s = oss.str();
        std::replace(s.begin(), s.end(), ',', ' ');

        // go through the data
        std::stringstream bufferStream;
        bufferStream.str(s);

        T value;
        #define NEXT_VALUE(name) if (!(bufferStream >> value)) { err_out = "Error reading <" name "> property."; return false; }

        // <lamp to luminaire geometry> <#tilt angles> <angles> <multiplying factors>
        if (ies_out.tilt == "INCLUDE") {
            NEXT_VALUE("lamp to luminaire geometry")    ies_out.lamp_to_luminaire_geometry = static_cast<int>(value);
            NEXT_VALUE("#tilt angles")                  ies_out.number_of_tilt_angles = static_cast<int>(value);

            ies_out.tilt_angles.reserve(ies_out.number_of_tilt_angles);
            ies_out.tilt_multiplying_factors.reserve(ies_out.number_of_tilt_angles);
            for (int i = 0; i < ies_out.number_of_tilt_angles; i++) {
                NEXT_VALUE("angles")        ies_out.tilt_angles.push_back(value);
            }
            for (int i = 0; i < ies_out.number_of_tilt_angles; i++) {
                NEXT_VALUE("angles")        ies_out.tilt_multiplying_factors.push_back(value);
            }
        }

        // <#lamps> <lumen/lamp> <multiplier> <#vertical angles> <#horizontal angles> <photometric type> <units type> <width> <length> <height>
        NEXT_VALUE("#lamps")                ies_out.number_lights = static_cast<int>(value);
        NEXT_VALUE("lumens/lamp")           ies_out.lumens_per_lamp = static_cast<int>(value);
        NEXT_VALUE("multiplier")            ies_out.multiplier = value;
        NEXT_VALUE("#vertical angles")      ies_out.number_vertical_angles = static_cast<int>(value);
        NEXT_VALUE("#horizontal angles")    ies_out.number_horizontal_angles = static_cast<int>(value);
        NEXT_VALUE("photometric type")      ies_out.photometric_type = static_cast<int>(value);
        NEXT_VALUE("units type")            ies_out.units_type = static_cast<int>(value);
        NEXT_VALUE("width")                 ies_out.width = value;
        NEXT_VALUE("length")                ies_out.length = value;
        NEXT_VALUE("height")                ies_out.height = value;
        // <ballast factor> <future use> <input watts>
        NEXT_VALUE("ballast factor")        ies_out.ballast_factor = value;
        NEXT_VALUE("future use")            ies_out.future_use = value;
        NEXT_VALUE("input watts")           ies_out.input_watts = value;
        // <vertical angles>
        ies_out.min_vertical_angle = 360;
        ies_out.max_vertical_angle = -360;
        ies_out.vertical_angles.reserve(ies_out.number_vertical_angles);
        for (int i = 0; i < ies_out.number_vertical_angles; i++) {
            NEXT_VALUE("vertical angles")   ies_out.vertical_angles.push_back(value);
            if (value < ies_out.min_vertical_angle) ies_out.min_vertical_angle = value;
            if (value > ies_out.max_vertical_angle) ies_out.max_vertical_angle = value;
        }
        // <horizontal angles>
        ies_out.min_horizontal_angle = 360;
        ies_out.max_horizontal_angle = -360;
        ies_out.horizontal_angles.reserve(ies_out.number_horizontal_angles);
        for (int i = 0; i < ies_out.number_horizontal_angles; i++) {
            NEXT_VALUE("horizontal angles") ies_out.horizontal_angles.push_back(value);
            if (value < ies_out.min_horizontal_angle) ies_out.min_horizontal_angle = value;
            if (value > ies_out.max_horizontal_angle) ies_out.max_horizontal_angle = value;
        }
        // <candela values for all vertical angles at first horizontal angle>
        //                                              :
        // <candela values for all vertical angles at last horizontal angle>
        ies_out.candela.reserve(static_cast<uint64_t>(ies_out.number_vertical_angles) * static_cast<uint64_t>(ies_out.number_horizontal_angles));
        ies_out.max_candela = 0;
        for (int i = 0; i < ies_out.number_vertical_angles * ies_out.number_horizontal_angles; i++) {
            NEXT_VALUE("candela values")    ies_out.candela.push_back(value);
            if (ies_out.max_candela < value) ies_out.max_candela = value;
        }
#undef NEXT_VALUE
        return true;
    }

    static std::string write_ies_to_buffer(const light& ies, const uint32_t precision = std::numeric_limits<T>::max_digits10) {
        std::stringstream ss;
        ss.precision(precision);

        /*
        ** HEADER
        */
        ss << "IESNA:" << ies.ies_version << std::endl;
        for (const auto& pair : ies.properties) ss << "[" << pair.first << "] " << pair.second << std::endl;

        /*
        ** DATA
        */
        ss << "TILT=" << ies.tilt << std::endl;
        if (ies.tilt == "INCLUDE") {
            // <lamp to luminaire geometry> <#tilt angles> <angles> <multiplying factors>
            ss << ies.lamp_to_luminaire_geometry << " " << ies.number_of_tilt_angles;
            for (int i = 0; i < ies.number_of_tilt_angles; i++) {
                ss << ies.tilt_angles[i];
                if (i < (ies.number_of_tilt_angles - 1)) ss << " ";
            }
            for (int i = 0; i < ies.number_of_tilt_angles; i++) {
                ss << ies.tilt_multiplying_factors[i];
                if (i < (ies.number_of_tilt_angles - 1)) ss << " ";
            }
            ss << std::endl;
        }

        // <#lamps> <lumen/lamp> <multiplier> <#vertical angles> <#horizontal angles> <photometric type> <units type> <width> <length> <height>
        ss << ies.number_lights << " " << ies.lumens_per_lamp << " " << ies.multiplier << " ";
        ss << ies.number_vertical_angles << " " << ies.number_horizontal_angles << " " << ies.photometric_type << " ";
        ss << ies.units_type << " " << ies.width << " " << ies.length << " " << ies.height << std::endl;
        // <ballast factor> <future use> <input watts>
        ss << ies.ballast_factor << " " << ies.future_use << " " << ies.input_watts << std::endl;
        // <vertical angles>
        for (int i = 0; i < ies.number_vertical_angles; i++) {
            ss << ies.vertical_angles[i];
            if (i < (ies.number_vertical_angles - 1)) ss << " ";
            else ss << std::endl;
        }
        // <horizontal angles>
        for (int i = 0; i < ies.number_horizontal_angles; i++) {
            ss << ies.horizontal_angles[i];
            if (i < (ies.number_horizontal_angles - 1)) ss << " ";
            else ss << std::endl;
        }
        // <candela values for all vertical angles at first horizontal angle>
        //                                              :
        // <candela values for all vertical angles at last horizontal angle>
        const int count = ies.number_vertical_angles * ies.number_horizontal_angles;
        for (int i = 0; i < count; i++) {
            ss << ies.candela[i];
            if (i != 0 && (i + 1) % (ies.number_vertical_angles) == 0) ss << std::endl;
            else if (i < (count - 1)) ss << " ";
        }

        return ss.str();
    }

private:
    /* read property in current line and return value */
    static bool read_property(const std::string& name, std::string& line, std::string& out) {
        size_t pos = line.find(name);
        if (pos == std::string::npos) return false;

        const std::string substring = line.substr(pos + name.size(), line.size());
        pos = substring.find_first_not_of(' ');
        if (pos == std::string::npos) return false;
        out = substring.substr(pos, substring.size());
        return true;
    }
    /* read property in current line and add it to the dict */
    static bool read_property(std::string& line, std::map<std::string, std::string>& property_map) {
        const std::string start_delim = "[";
        const std::string stop_delim = "]";
        const size_t attribute_start = line.find(start_delim) + start_delim.size();
        const size_t attribute_end = line.find_last_of(stop_delim);
        if (attribute_start == std::string::npos || attribute_end == std::string::npos) return false;

        std::string attribute = line.substr(attribute_start, attribute_end - attribute_start);
        std::string property;
        if (!read_property(attribute + stop_delim, line, property)) return false;
        property_map.insert({ attribute, property });
        return true;
    }
};