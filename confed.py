"""Country -> FIFA confederation map (martj42 dataset team names)."""

CONFEDERATIONS = {
    "CONMEBOL": ["Argentina", "Brazil", "Uruguay", "Colombia", "Chile", "Peru", "Paraguay", "Ecuador", "Bolivia", "Venezuela"],
    "UEFA": ["Spain", "France", "England", "Germany", "Italy", "Netherlands", "Portugal", "Belgium", "Croatia", "Denmark",
             "Switzerland", "Austria", "Ukraine", "Sweden", "Poland", "Wales", "Scotland", "Serbia", "Turkey", "Czech Republic",
             "Hungary", "Norway", "Republic of Ireland", "Russia", "Romania", "Slovakia", "Slovenia", "Greece", "Finland",
             "Iceland", "Bosnia and Herzegovina", "North Macedonia", "Northern Ireland", "Albania", "Montenegro", "Bulgaria",
             "Georgia", "Israel", "Luxembourg", "Belarus", "Kosovo", "Armenia", "Azerbaijan", "Cyprus", "Estonia", "Kazakhstan",
             "Latvia", "Lithuania", "Moldova", "Malta", "Faroe Islands", "Andorra", "Gibraltar", "San Marino", "Liechtenstein"],
    "CONCACAF": ["United States", "Mexico", "Canada", "Costa Rica", "Jamaica", "Panama", "Honduras", "El Salvador",
                 "Trinidad and Tobago", "Haiti", "Guatemala", "Curaçao", "Cuba", "Nicaragua", "Suriname", "Guadeloupe",
                 "Martinique", "Saint Kitts and Nevis", "Antigua and Barbuda", "Grenada", "Belize", "Dominican Republic",
                 "Barbados", "Bermuda", "French Guiana", "Cayman Islands", "Aruba", "Saint Lucia",
                 "Saint Vincent and the Grenadines", "Puerto Rico", "Dominica", "Bahamas", "Guyana", "Montserrat",
                 "Turks and Caicos Islands", "British Virgin Islands", "US Virgin Islands", "Anguilla", "Sint Maarten", "Bonaire"],
    "CAF": ["Morocco", "Senegal", "Tunisia", "Algeria", "Egypt", "Nigeria", "Cameroon", "Ghana", "Ivory Coast", "Mali",
            "Burkina Faso", "DR Congo", "South Africa", "Cape Verde", "Guinea", "Zambia", "Uganda", "Benin", "Gabon",
            "Equatorial Guinea", "Angola", "Mozambique", "Madagascar", "Kenya", "Mauritania", "Namibia", "Congo", "Togo",
            "Sudan", "Zimbabwe", "Tanzania", "Comoros", "Guinea-Bissau", "Libya", "Ethiopia", "Sierra Leone", "Niger",
            "Malawi", "Rwanda", "Central African Republic", "Liberia", "Botswana", "Burundi", "Lesotho", "Eswatini",
            "Gambia", "Chad", "South Sudan", "Somalia", "Seychelles", "Mauritius", "São Tomé and Príncipe", "Djibouti", "Eritrea"],
    "AFC": ["Japan", "South Korea", "Iran", "Australia", "Saudi Arabia", "Qatar", "Iraq", "United Arab Emirates", "Uzbekistan",
            "China PR", "Jordan", "Oman", "Bahrain", "Syria", "Lebanon", "Palestine", "India", "Vietnam", "Thailand", "Kuwait",
            "Tajikistan", "Kyrgyzstan", "Turkmenistan", "North Korea", "Hong Kong", "Indonesia", "Malaysia", "Philippines",
            "Myanmar", "Singapore", "Yemen", "Afghanistan", "Maldives", "Nepal", "Bangladesh", "Sri Lanka", "Cambodia",
            "Mongolia", "Bhutan", "Laos", "Brunei", "Pakistan", "Guam", "Macau", "Timor-Leste", "Chinese Taipei"],
    "OFC": ["New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands", "Vanuatu", "Tahiti", "New Caledonia", "Samoa",
            "Tonga", "Cook Islands", "American Samoa"],
}

_TEAM_CONFED = {t: c for c, teams in CONFEDERATIONS.items() for t in teams}


def confed_of(team):
    return _TEAM_CONFED.get(team, "OTHER")
