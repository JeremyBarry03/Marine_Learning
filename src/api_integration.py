"""Placeholder for future marine data integrations."""

SPECIES_METADATA = {
    "Scallop": {
        "latin_name": "Pectinidae spp.",
        "habitat": "Sandy and gravel seafloors",
        "notes": "Commercially important filter feeders."
    },
    "Roundfish": {
        "latin_name": "Various demersal genera",
        "habitat": "Continental shelf regions",
        "notes": "Represents round-bodied demersal finfish grouped for surveys."
    },
    "Crab": {
        "latin_name": "Brachyura infraorder",
        "habitat": "Benthic zones, crevices, rocky substrates",
        "notes": "Decapod crustaceans with carapace-protected bodies."
    },
    "Whelk": {
        "latin_name": "Buccinidae family",
        "habitat": "Cold-water soft sediments",
        "notes": "Predatory gastropods attracted to baited camera deployments."
    },
    "Skate": {
        "latin_name": "Rajidae family",
        "habitat": "Soft bottoms, continental shelf, slope",
        "notes": "Cartilaginous fish using camouflaged coloration."
    },
    "Flatfish": {
        "latin_name": "Pleuronectiformes order",
        "habitat": "Soft substrates",
        "notes": "Laterally compressed fish with both eyes on one side."
    },
    "Eel": {
        "latin_name": "Anguilliformes order",
        "habitat": "Burrows and crevices",
        "notes": "Elongated bodies suitable for crevice dwelling."
    },
}


def get_species_info(species: str) -> dict:
    """
    Returns static metadata for the requested species.

    Placeholder for future integrations with marine knowledge bases or APIs.
    """
    return SPECIES_METADATA.get(species, {})


if __name__ == "__main__":
    for name in SPECIES_METADATA:
        print(f"{name}: {SPECIES_METADATA[name]['notes']}")
