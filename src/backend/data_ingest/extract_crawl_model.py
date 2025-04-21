from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class PartyEnum(Enum):
    """Enumeration of political parties in Singapore."""
    PAP = "People's Action Party"
    WP = "Workers' Party"
    PSP = "Progress Singapore Party"
    SDP = "Singapore Democratic Party"
    RP = "Reform Party"
    SPP = "Singapore People's Party"
    NSP = "National Solidarity Party"
    PV = "Peoples Voice"


class Candidate(BaseModel):
    """Model for candidate information."""
    name: str
    party: str
    constituency: str
    bio: Optional[str] = None


class ExtractedParty(BaseModel):
    """Model for extracted party information."""
    name: str
    candidates: List[Candidate] = []
    additional_info: Optional[str] = None


class ExtractedConstituency(BaseModel):
    """Model for extracted data for 1 constituency."""
    name: str
    constituency_type: str
    contesting_parties: List[str] = []
    candidates: List[Candidate] = []


class ExtractedElectionData(BaseModel):
    """Model for extracted data for a list of constituencies."""
    constituencies: List[ExtractedConstituency] = []

