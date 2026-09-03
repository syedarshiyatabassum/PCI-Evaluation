import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import json
import math


# ============================================================
# SETTINGS
# ============================================================

RESULT_FOLDER = "RESULTS"

# ------------------------------------------------------------
# GRAPH CROP
# ------------------------------------------------------------
# Change these only if your graph is located differently.

CROP_Y1 = 55
CROP_Y2 = 275
CROP_X1 = 50
CROP_X2 = 395


# ------------------------------------------------------------
# ASTM GRAPH X-AXIS LIMITS
# ------------------------------------------------------------
# log10(Density)
#
# Example:
# Density = 0.1  -> log10 = -1
# Density = 1    -> log10 = 0
# Density = 10   -> log10 = 1
# Density = 100  -> log10 = 2
#
# Change these if your ASTM graph has different limits.

LOG_DENSITY_MIN = -1.0
LOG_DENSITY_MAX = 2.0


# ------------------------------------------------------------
# DEDUCT VALUE AXIS
# ------------------------------------------------------------

DV_MIN = 0.0
DV_MAX = 100.0


# ------------------------------------------------------------
# CURVE PROCESSING
# ------------------------------------------------------------

SMOOTH_WINDOW = 7

MIN_COMPONENT_AREA = 8


# ------------------------------------------------------------
# POLYNOMIAL DEGREES TO TEST
# ------------------------------------------------------------

POLYNOMIAL_DEGREES = [2, 3, 4, 5]


# ============================================================
# DISTRESS TYPES
# ============================================================

DISTRESSES = [

    "Alligator Cracking",

    "Bleeding",

    "Rutting",

    "Swell",

    "Bumps and Sags",

    "Potholes",

    "Patching"

]


# ============================================================
# SEVERITY LEVELS
# ============================================================

SEVERITIES = [

    "Low",

    "Medium",

    "High"

]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

INPUT_IMAGES = []

GENERATED_EQUATIONS = {}


# ============================================================
# CREATE RESULT FOLDER
# ============================================================

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

def header(text):

    print("\n")
    print("=" * 75)
    print(text)
    print("=" * 75)


# ============================================================
# STEP 1
# SELECT ALL IMAGES
# ============================================================

def select_images():

    global INPUT_IMAGES

    header(
        "STEP 1 - INPUT ALL ASTM GRAPH IMAGES"
    )

    print(
        "\nSelect ALL the ASTM graph images."
    )

    print(
        "You can select multiple images at the same time."
    )

    # --------------------------------------------------------
    # OPEN FILE SELECTION WINDOW
    # --------------------------------------------------------

    root = tk.Tk()

    root.withdraw()

    files = filedialog.askopenfilenames(

        title="Select ASTM Graph Images",

        filetypes=[

            (
                "Image files",
                "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
            ),

            (
                "PNG files",
                "*.png"
            ),

            (
                "JPG files",
                "*.jpg *.jpeg"
            ),

            (
                "All files",
                "*.*"
            )

        ]

    )

    root.destroy()


    if not files:

        print(
            "\nNo images selected."
        )

        return


    INPUT_IMAGES = []


    # ========================================================
    # ASK DISTRESS AND SEVERITY FOR EVERY IMAGE
    # ========================================================

    for i, file_path in enumerate(
        files,
        start=1
    ):

        print("\n")
        print("-" * 75)

        print(
            f"IMAGE {i} OF {len(files)}"
        )

        print(
            "File:",
            os.path.basename(file_path)
        )

        print("-" * 75)


        # ----------------------------------------------------
        # DISTRESS TYPE
        # ----------------------------------------------------

        print(
            "\nSelect Distress Type:"
        )

        for j, distress in enumerate(
            DISTRESSES,
            start=1
        ):

            print(
                f"{j}. {distress}"
            )


        while True:

            try:

                choice = int(
                    input(
                        "\nEnter distress number: "
                    )
                )

                if (
                    1 <= choice <= len(DISTRESSES)
                ):

                    distress = DISTRESSES[
                        choice - 1
                    ]

                    break

            except:

                pass


            print(
                "Invalid choice. Try again."
            )


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        print(
            "\nSelect Severity:"
        )

        for j, severity in enumerate(
            SEVERITIES,
            start=1
        ):

            print(
                f"{j}. {severity}"
            )


        while True:

            try:

                choice = int(
                    input(
                        "\nEnter severity number: "
                    )
                )

                if (
                    1 <= choice <= len(SEVERITIES)
                ):

                    severity = SEVERITIES[
                        choice - 1
                    ]

                    break

            except:

                pass


            print(
                "Invalid choice. Try again."
            )


        # ----------------------------------------------------
        # STORE INFORMATION
        # ----------------------------------------------------

        INPUT_IMAGES.append({

            "file": file_path,

            "distress": distress,

            "severity": severity

        })


        print(
            "\nImage registered successfully."
        )

        print(
            "Distress:",
            distress
        )

        print(
            "Severity:",
            severity
        )


    # ========================================================
    # SAVE INPUT IMAGE INFORMATION
    # ========================================================

    input_json = os.path.join(
        RESULT_FOLDER,
        "input_images.json"
    )


    with open(
        input_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            INPUT_IMAGES,
            file,
            indent=4
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    header(
        "ALL INPUT IMAGES"
    )


    for i, item in enumerate(
        INPUT_IMAGES,
        start=1
    ):

        print(

            f"{i}. "
            f"{os.path.basename(item['file'])}"
            f" -> "
            f"{item['distress']}"
            f" -> "
            f"{item['severity']}"

        )


    print(
        "\nTotal images:",
        len(INPUT_IMAGES)
    )


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(binary):

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )


    cleaned = np.zeros_like(
        binary
    )


    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]


        if area >= MIN_COMPONENT_AREA:

            cleaned[
                labels == i
            ] = 255


    return cleaned


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    file_path
):

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    img = cv2.imread(
        file_path
    )


    if img is None:

        print(
            "ERROR: Image could not be read."
        )

        return None


    # --------------------------------------------------------
    # CROP GRAPH
    # --------------------------------------------------------

    crop = img[
        CROP_Y1:CROP_Y2,
        CROP_X1:CROP_X2
    ]


    if crop.size == 0:

        print(
            "ERROR: Invalid crop."
        )

        return None


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8)

    )


    enhanced = clahe.apply(
        gray
    )


    # --------------------------------------------------------
    # ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(

        enhanced,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        21,

        5

    )


    # ========================================================
    # HORIZONTAL GRID
    # ========================================================

    horizontal_kernel = \
        cv2.getStructuringElement(

            cv2.MORPH_RECT,

            (35, 1)

        )


    horizontal = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        horizontal_kernel

    )


    # ========================================================
    # VERTICAL GRID
    # ========================================================

    vertical_kernel = \
        cv2.getStructuringElement(

            cv2.MORPH_RECT,

            (1, 35)

        )


    vertical = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        vertical_kernel

    )


    # ========================================================
    # REMOVE GRID
    # ========================================================

    grid = cv2.bitwise_or(

        horizontal,

        vertical

    )


    curve_mask = cv2.subtract(

        binary,

        grid

    )


    # ========================================================
    # MORPHOLOGICAL CLEANING
    # ========================================================

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3, 3)

    )


    curve_mask = cv2.morphologyEx(

        curve_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    curve_mask = cv2.morphologyEx(

        curve_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    # ========================================================
    # REMOVE SMALL COMPONENTS
    # ========================================================

    curve_mask = remove_small_components(

        curve_mask

    )


    # ========================================================
    # CURVE EXTRACTION
    #
    # We follow a continuous curve instead of simply
    # selecting the middle pixel.
    # ========================================================

    height, width = curve_mask.shape


    points_x = []

    points_y = []


    previous_y = None


    for x in range(width):

        y_pixels = np.where(

            curve_mask[:, x] > 0

        )[0]


        if len(y_pixels) == 0:

            continue


        # ----------------------------------------------------
        # FIRST POINT
        # ----------------------------------------------------

        if previous_y is None:

            y = np.median(
                y_pixels
            )


        else:

            # ------------------------------------------------
            # Find pixel closest to previous curve point
            # ------------------------------------------------

            distances = np.abs(

                y_pixels
                -
                previous_y

            )


            index = np.argmin(
                distances
            )


            y = y_pixels[
                index
            ]


        # ----------------------------------------------------
        # REJECT VERY LARGE JUMPS
        # ----------------------------------------------------

        if previous_y is not None:

            if abs(
                y - previous_y
            ) > 25:

                local_pixels = y_pixels[
                    np.abs(
                        y_pixels
                        -
                        previous_y
                    ) <= 25
                ]


                if len(
                    local_pixels
                ) == 0:

                    continue


                y = local_pixels[
                    np.argmin(
                        np.abs(
                            local_pixels
                            -
                            previous_y
                        )
                    )
                ]


        points_x.append(
            x
        )

        points_y.append(
            y
        )


        previous_y = y


    # --------------------------------------------------------
    # CHECK POINTS
    # --------------------------------------------------------

    if len(points_x) < 20:

        print(
            "ERROR: Not enough curve points found."
        )

        return None


    points_x = np.array(
        points_x,
        dtype=float
    )


    points_y = np.array(
        points_y,
        dtype=float
    )


    # ========================================================
    # MEDIAN SMOOTHING
    # ========================================================

    smooth_y = points_y.copy()

    w = SMOOTH_WINDOW


    for i in range(

        w,

        len(points_y) - w

    ):

        smooth_y[i] = np.median(

            points_y[
                i-w:i+w+1
            ]

        )


    points_y = smooth_y


    # ========================================================
    # PIXEL X -> ASTM LOG DENSITY
    # ========================================================

    log_density = (

        LOG_DENSITY_MIN

        +

        (
            points_x
            /
            (width - 1)
        )

        *

        (
            LOG_DENSITY_MAX
            -
            LOG_DENSITY_MIN
        )

    )


    # ========================================================
    # PIXEL Y -> DEDUCT VALUE
    # ========================================================

    deduct = (

        DV_MAX

        -

        (
            points_y
            /
            (height - 1)
        )

        *

        (
            DV_MAX
            -
            DV_MIN
        )

    )


    # ========================================================
    # LIMIT DV
    # ========================================================

    deduct = np.clip(

        deduct,

        DV_MIN,

        DV_MAX

    )


    # ========================================================
    # OUTLIER REMOVAL
    # ========================================================

    if len(deduct) > 20:

        median = np.median(
            deduct
        )


        mad = np.median(

            np.abs(

                deduct
                -
                median

            )

        )


        if mad > 0:

            robust_z = (

                np.abs(

                    deduct
                    -
                    median

                )

                /

                (
                    1.4826
                    *
                    mad
                )

            )


            keep = (
                robust_z < 6
            )


            points_x = points_x[
                keep
            ]

            points_y = points_y[
                keep
            ]

            log_density = \
                log_density[
                    keep
                ]

            deduct = \
                deduct[
                    keep
                ]


    # ========================================================
    # RETURN DATA
    # ========================================================

    return {

        "log_density":
            log_density,

        "deduct":
            deduct,

        "pixel_x":
            points_x,

        "pixel_y":
            points_y,

        "curve_mask":
            curve_mask,

        "crop":
            crop

    }


# ============================================================
# COMBINE MULTIPLE IMAGES
# ============================================================

def combine_curves(
    results
):

    if len(results) == 0:

        return None, None


    # --------------------------------------------------------
    # COMMON X VALUES
    # --------------------------------------------------------

    common_x = np.linspace(

        LOG_DENSITY_MIN,

        LOG_DENSITY_MAX,

        800

    )


    all_curves = []


    # ========================================================
    # INTERPOLATE EACH IMAGE
    # ========================================================

    for result in results:

        x = result[
            "log_density"
        ]

        y = result[
            "deduct"
        ]


        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        order = np.argsort(
            x
        )


        x = x[
            order
        ]

        y = y[
            order
        ]


        # ----------------------------------------------------
        # REMOVE DUPLICATE X VALUES
        # ----------------------------------------------------

        unique_x, indices = \
            np.unique(

                x,

                return_index=True

            )


        unique_y = y[
            indices
        ]


        if len(unique_x) < 5:

            continue


        # ----------------------------------------------------
        # INTERPOLATE
        # ----------------------------------------------------

        interpolated = np.interp(

            common_x,

            unique_x,

            unique_y

        )


        all_curves.append(
            interpolated
        )


    if len(all_curves) == 0:

        return None, None


    # --------------------------------------------------------
    # MEDIAN OF MULTIPLE IMAGES
    # --------------------------------------------------------

    combined_y = np.median(

        np.array(
            all_curves
        ),

        axis=0

    )


    return (

        common_x,

        combined_y

    )


# ============================================================
# CALCULATE R2
# ============================================================

def calculate_r2(
    actual,
    predicted
):

    ss_res = np.sum(

        (
            actual
            -
            predicted
        ) ** 2

    )


    ss_tot = np.sum(

        (
            actual
            -
            np.mean(actual)
        ) ** 2

    )


    if ss_tot == 0:

        return 0


    return (

        1
        -
        (
            ss_res
            /
            ss_tot
        )

    )


# ============================================================
# CALCULATE RMSE
# ============================================================

def calculate_rmse(
    actual,
    predicted
):

    return np.sqrt(

        np.mean(

            (
                actual
                -
                predicted
            ) ** 2

        )

    )


# ============================================================
# FIND BEST POLYNOMIAL
# ============================================================

def find_best_polynomial(
    x,
    y
):

    best = None


    # ========================================================
    # TRY DEGREE 2, 3, 4 AND 5
    # ========================================================

    for degree in POLYNOMIAL_DEGREES:

        try:

            coefficients = np.polyfit(

                x,

                y,

                degree

            )


            polynomial = np.poly1d(

                coefficients

            )


            predicted = polynomial(
                x
            )


            r2 = calculate_r2(

                y,

                predicted

            )


            rmse = calculate_rmse(

                y,

                predicted

            )


            # ------------------------------------------------
            # ADJUSTED R2
            # ------------------------------------------------

            n = len(y)

            p = degree


            if (
                n > p + 1
                and
                r2 < 1
            ):

                adjusted_r2 = (

                    1

                    -

                    (

                        (
                            1 - r2
                        )

                        *

                        (
                            n - 1
                        )

                        /

                        (
                            n - p - 1
                        )

                    )

                )

            else:

                adjusted_r2 = r2


            candidate = {

                "degree":
                    degree,

                "coefficients":
                    coefficients,

                "polynomial":
                    polynomial,

                "predicted":
                    predicted,

                "r2":
                    r2,

                "adjusted_r2":
                    adjusted_r2,

                "rmse":
                    rmse

            }


            # ------------------------------------------------
            # SELECT BEST
            # ------------------------------------------------

            if best is None:

                best = candidate

            elif (

                candidate[
                    "adjusted_r2"
                ]

                >

                best[
                    "adjusted_r2"
                ]

            ):

                best = candidate


        except Exception:

            continue


    return best


# ============================================================
# FORMAT EQUATION
#
# IMPORTANT:
# ONLY ASCII CHARACTERS ARE USED.
# This prevents Windows UnicodeEncodeError.
# ============================================================

def format_equation(
    coefficients
):

    degree = len(
        coefficients
    ) - 1


    equation = ""


    for i, coefficient in enumerate(
        coefficients
    ):

        power = degree - i


        value = abs(
            coefficient
        )


        # ----------------------------------------------------
        # Ignore extremely small values
        # ----------------------------------------------------

        if value < 1e-10:

            continue


        # ----------------------------------------------------
        # SIGN
        # ----------------------------------------------------

        if coefficient >= 0:

            sign = "+"

        else:

            sign = "-"


        # ----------------------------------------------------
        # TERM
        # ----------------------------------------------------

        if power == 0:

            term = (
                f"{value:.8f}"
            )


        elif power == 1:

            term = (
                f"{value:.8f}*x"
            )


        else:

            term = (
                f"{value:.8f}*x^{power}"
            )


        # ----------------------------------------------------
        # FIRST TERM
        # ----------------------------------------------------

        if equation == "":

            if coefficient < 0:

                equation = (
                    "- " + term
                )

            else:

                equation = term


        # ----------------------------------------------------
        # OTHER TERMS
        # ----------------------------------------------------

        else:

            equation += (

                f" {sign} {term}"

            )


    return equation


# ============================================================
# STEP 2
# GENERATE ALL EQUATIONS
# ============================================================

def generate_equations():

    global GENERATED_EQUATIONS


    header(
        "STEP 2 - GENERATE ALL EQUATIONS"
    )


    if len(INPUT_IMAGES) == 0:

        print(
            "\nNo images found."
        )

        print(
            "Please run Step 1 first."
        )

        return {}


    # ========================================================
    # GROUP IMAGES BY DISTRESS + SEVERITY
    # ========================================================

    groups = {}


    for item in INPUT_IMAGES:

        key = (

            item["distress"],

            item["severity"]

        )


        if key not in groups:

            groups[key] = []


        groups[key].append(
            item["file"]
        )


    equations = {}


    # ========================================================
    # PROCESS EACH GROUP
    # ========================================================

    for (

        distress,

        severity

    ), files in groups.items():


        print("\n")
        print("=" * 75)

        print(
            distress,
            "-",
            severity
        )

        print("=" * 75)


        processed_images = []


        # ----------------------------------------------------
        # PROCESS EACH IMAGE
        # ----------------------------------------------------

        for file_path in files:

            print(

                "\nProcessing:",
                os.path.basename(
                    file_path
                )

            )


            result = process_image(
                file_path
            )


            if result is not None:

                processed_images.append(
                    result
                )


        if len(
            processed_images
        ) == 0:

            print(
                "No usable images."
            )

            continue


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        x, y = combine_curves(

            processed_images

        )


        if x is None:

            print(
                "Could not combine curves."
            )

            continue


        # ----------------------------------------------------
        # FIND BEST POLYNOMIAL
        # ----------------------------------------------------

        best = find_best_polynomial(

            x,

            y

        )


        if best is None:

            print(
                "Polynomial fitting failed."
            )

            continue


        coefficients = \
            best["coefficients"]


        equation = format_equation(

            coefficients

        )


        # ====================================================
        # STORE
        # ====================================================

        if distress not in equations:

            equations[
                distress
            ] = {}


        equations[
            distress
        ][severity] = {

            "degree":
                int(
                    best["degree"]
                ),

            "coefficients":
                coefficients.tolist(),

            "equation":
                equation,

            "r2":
                float(
                    best["r2"]
                ),

            "adjusted_r2":
                float(
                    best["adjusted_r2"]
                ),

            "rmse":
                float(
                    best["rmse"]
                )

        }


        # ====================================================
        # DISPLAY
        # ====================================================

        print(
            "\nBEST POLYNOMIAL"
        )

        print(
            "Degree:",
            best["degree"]
        )


        print(
            "\nEquation:"
        )

        print(
            "DV =",
            equation
        )


        print(
            "\nR2 =",
            round(
                best["r2"],
                6
            )
        )


        print(
            "Adjusted R2 =",
            round(
                best["adjusted_r2"],
                6
            )
        )


        print(
            "RMSE =",
            round(
                best["rmse"],
                6
            )
        )


        # ====================================================
        # PLOT
        # ====================================================

        plt.figure(
            figsize=(9, 6)
        )


        plt.scatter(

            x,

            y,

            s=7,

            label="Extracted ASTM curve"

        )


        fitted = best[
            "polynomial"
        ](x)


        plt.plot(

            x,

            fitted,

            linewidth=2,

            label=(
                "Best polynomial "
                f"degree {best['degree']}"
            )

        )


        plt.xlabel(
            "log10(Density)"
        )


        plt.ylabel(
            "Deduct Value"
        )


        plt.title(

            distress
            + " - "
            + severity

        )


        plt.grid(
            True
        )


        plt.legend()


        safe_name = (

            distress
            .replace(
                " ",
                "_"
            )

        )


        plot_path = os.path.join(

            RESULT_FOLDER,

            safe_name
            + "_"
            + severity
            + "_FIT.png"

        )


        plt.savefig(

            plot_path,

            dpi=400,

            bbox_inches="tight"

        )


        plt.close()


    # ========================================================
    # STORE GLOBALLY
    # ========================================================

    GENERATED_EQUATIONS = equations


    # ========================================================
    # DISPLAY ALL EQUATIONS
    # ========================================================

    header(
        "ALL EQUATIONS GENERATED"
    )


    if len(equations) == 0:

        print(
            "No equations generated."
        )

        return equations


    for distress in equations:

        for severity in equations[
            distress
        ]:

            data = equations[
                distress
            ][severity]


            print(
                f"\n{distress} - {severity}"
            )


            print(
                "Equation:"
            )


            print(
                "DV =",
                data["equation"]
            )


            print(
                "R2 =",
                round(
                    data["r2"],
                    6
                )
            )


    return equations


# ============================================================
# STEP 3
# SAVE ALL EQUATIONS
# ============================================================

def save_equations():

    header(
        "STEP 3 - SAVE ALL EQUATIONS"
    )


    if not GENERATED_EQUATIONS:

        print(
            "\nNo equations generated."
        )

        print(
            "Please run Step 2 first."
        )

        return


    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(

        RESULT_FOLDER,

        "ALL_EQUATIONS.json"

    )


    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            GENERATED_EQUATIONS,

            file,

            indent=4

        )


    # ========================================================
    # SAVE TEXT
    # ========================================================

    txt_path = os.path.join(

        RESULT_FOLDER,

        "ALL_EQUATIONS.txt"

    )


    with open(

        txt_path,

        "w",

        encoding="utf-8"

    ) as file:


        file.write(
            "ASTM POLYNOMIAL EQUATIONS\n"
        )


        file.write(
            "=" * 75
            + "\n\n"
        )


        for distress in GENERATED_EQUATIONS:

            for severity in GENERATED_EQUATIONS[
                distress
            ]:


                data = GENERATED_EQUATIONS[
                    distress
                ][severity]


                file.write(

                    f"Distress: {distress}\n"

                )


                file.write(

                    f"Severity: {severity}\n"

                )


                file.write(

                    f"Polynomial Degree: "
                    f"{data['degree']}\n"

                )


                file.write(

                    f"Equation: "
                    f"DV = "
                    f"{data['equation']}\n"

                )


                file.write(

                    f"R2: "
                    f"{data['r2']:.8f}\n"

                )


                file.write(

                    f"Adjusted R2: "
                    f"{data['adjusted_r2']:.8f}\n"

                )


                file.write(

                    f"RMSE: "
                    f"{data['rmse']:.8f}\n"

                )


                file.write(
                    "-" * 75
                    + "\n\n"
                )


    # ========================================================
    # SAVE SUCCESS MESSAGE
    # ========================================================

    print(
        "\nALL EQUATIONS SAVED SUCCESSFULLY."
    )


    print(
        "\nText file:"
    )


    print(
        txt_path
    )


    print(
        "\nJSON file:"
    )


    print(
        json_path
    )


# ============================================================
# LOAD SAVED EQUATIONS
# ============================================================

def load_equations():

    path = os.path.join(

        RESULT_FOLDER,

        "ALL_EQUATIONS.json"

    )


    if not os.path.exists(
        path
    ):

        return None


    try:

        with open(

            path,

            "r",

            encoding="utf-8"

        ) as file:

            return json.load(
                file
            )

    except:

        return None


# ============================================================
# STEP 4
# CALCULATE PCI
# ============================================================

def calculate_pci():

    header(
        "STEP 4 - CALCULATE PCI USING DENSITY"
    )


    # ========================================================
    # GET EQUATIONS
    # ========================================================

    equations = GENERATED_EQUATIONS


    if not equations:

        equations = load_equations()


    if not equations:

        print(
            "\nNo equations found."
        )

        print(
            "Please generate and save equations first."
        )

        return


    # ========================================================
    # NUMBER OF DISTRESS ENTRIES
    # ========================================================

    while True:

        try:

            n = int(

                input(

                    "\nEnter number of "
                    "distress entries: "

                )

            )


            if n > 0:

                break


        except:

            pass


        print(
            "Enter a valid number."
        )


    deduct_values = []


    # ========================================================
    # ENTER EACH DISTRESS
    # ========================================================

    for i in range(n):


        print("\n")
        print("-" * 75)

        print(
            f"DISTRESS {i + 1}"
        )

        print("-" * 75)


        # ====================================================
        # DISTRESS TYPE
        # ====================================================

        available_distresses = list(

            equations.keys()

        )


        print(
            "\nAvailable distress types:"
        )


        for j, distress in enumerate(

            available_distresses,

            start=1

        ):

            print(
                f"{j}. {distress}"
            )


        while True:

            try:

                choice = int(

                    input(
                        "\nEnter number: "
                    )

                )


                if (

                    1
                    <= choice
                    <= len(
                        available_distresses
                    )

                ):

                    distress = \
                        available_distresses[
                            choice - 1
                        ]

                    break


            except:

                pass


            print(
                "Invalid choice."
            )


        # ====================================================
        # SEVERITY
        # ====================================================

        available_severities = list(

            equations[
                distress
            ].keys()

        )


        print(
            "\nAvailable severity levels:"
        )


        for j, severity in enumerate(

            available_severities,

            start=1

        ):

            print(
                f"{j}. {severity}"
            )


        while True:

            try:

                choice = int(

                    input(
                        "\nEnter number: "
                    )

                )


                if (

                    1
                    <= choice
                    <= len(
                        available_severities
                    )

                ):

                    severity = \
                        available_severities[
                            choice - 1
                        ]

                    break


            except:

                pass


            print(
                "Invalid choice."
            )


        # ====================================================
        # DENSITY
        # ====================================================

        while True:

            try:

                density = float(

                    input(
                        "\nEnter Density: "
                    )

                )


                if density > 0:

                    break


            except:

                pass


            print(
                "Density must be greater than zero."
            )


        # ====================================================
        # LOG DENSITY
        # ====================================================

        x = math.log10(
            density
        )


        # ====================================================
        # GET EQUATION COEFFICIENTS
        # ====================================================

        coefficients = np.array(

            equations[
                distress
            ][severity][
                "coefficients"
            ]

        )


        # ====================================================
        # CALCULATE DEDUCT VALUE
        # ====================================================

        dv = np.polyval(

            coefficients,

            x

        )


        # ----------------------------------------------------
        # LIMIT DV BETWEEN 0 AND 100
        # ----------------------------------------------------

        dv = max(

            0,

            min(

                100,

                dv

            )

        )


        deduct_values.append(
            dv
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        print(
            "\nEquation used:"
        )


        print(

            "DV =",

            equations[
                distress
            ][severity][
                "equation"
            ]

        )


        print(
            "\nlog10(Density) =",
            round(
                x,
                6
            )
        )


        print(
            "Deduct Value =",
            round(
                dv,
                3
            )
        )


    # ========================================================
    # TOTAL DEDUCT VALUE
    # ========================================================

    tdv = sum(
        deduct_values
    )


    # ========================================================
    # CORRECTED DEDUCT VALUE
    #
    # NOTE:
    # This follows the calculation method in your
    # previous code.
    # ========================================================

    cdv = sum(

        dv

        for dv in deduct_values

        if dv > 2

    )


    # ========================================================
    # PCI
    # ========================================================

    pci = 100 - cdv


    if pci < 0:

        pci = 0


    # ========================================================
    # CONDITION
    # ========================================================

    if pci >= 85:

        condition = "Excellent"

    elif pci >= 70:

        condition = "Very Good"

    elif pci >= 55:

        condition = "Good"

    elif pci >= 40:

        condition = "Fair"

    elif pci >= 25:

        condition = "Poor"

    elif pci >= 10:

        condition = "Very Poor"

    else:

        condition = "Failed"


    # ========================================================
    # FINAL RESULT
    # ========================================================

    header(
        "FINAL PCI RESULT"
    )


    print(
        "\nIndividual Deduct Values:"
    )


    for i, dv in enumerate(

        deduct_values,

        start=1

    ):

        print(

            f"DV {i} = "
            f"{dv:.3f}"

        )


    print(
        "\n"
        + "-" * 75
    )


    print(

        "Total Deduct Value (TDV) =",

        round(
            tdv,
            3
        )

    )


    print(

        "Corrected Deduct Value (CDV) =",

        round(
            cdv,
            3
        )

    )


    print(

        "PCI =",

        round(
            pci,
            3
        )

    )


    print(

        "Condition =",

        condition

    )


    print(
        "-" * 75
    )


    # ========================================================
    # SAVE PCI RESULT
    # ========================================================

    result_path = os.path.join(

        RESULT_FOLDER,

        "PCI_RESULTS.txt"

    )


    with open(

        result_path,

        "a",

        encoding="utf-8"

    ) as file:


        file.write(
            "\n"
            + "=" * 75
            + "\n"
        )


        file.write(
            "PCI CALCULATION\n"
        )


        file.write(
            "=" * 75
            + "\n"
        )


        file.write(

            f"TDV = {tdv:.3f}\n"

        )


        file.write(

            f"CDV = {cdv:.3f}\n"

        )


        file.write(

            f"PCI = {pci:.3f}\n"

        )


        file.write(

            f"Condition = {condition}\n"

        )


    print(
        "\nPCI result saved to:"
    )


    print(
        result_path
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    global GENERATED_EQUATIONS


    header(
        "AUTOMATIC ASTM CURVE + PCI PROGRAM"
    )


    while True:


        print("\n")

        print("=" * 75)

        print(
            "1. Input all ASTM graph images"
        )

        print(
            "2. Generate all equations"
        )

        print(
            "3. Save all equations"
        )

        print(
            "4. Calculate PCI using density"
        )

        print(
            "5. Exit"
        )

        print("=" * 75)


        choice = input(

            "\nEnter your choice: "

        )


        # ====================================================
        # OPTION 1
        # ====================================================

        if choice == "1":

            select_images()


        # ====================================================
        # OPTION 2
        # ====================================================

        elif choice == "2":

            GENERATED_EQUATIONS = \
                generate_equations()


        # ====================================================
        # OPTION 3
        # ====================================================

        elif choice == "3":

            save_equations()


        # ====================================================
        # OPTION 4
        # ====================================================

        elif choice == "4":

            calculate_pci()


        # ====================================================
        # OPTION 5
        # ====================================================

        elif choice == "5":

            print(
                "\nProgram closed."
            )

            break


        else:

            print(
                "\nInvalid choice. Please enter 1-5."
            )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()