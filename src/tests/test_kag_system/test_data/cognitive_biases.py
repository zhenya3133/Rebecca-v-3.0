"""
Тестовые данные по когнитивным искажениям для KAG Testing Framework
"""

COGNITIVE_BIASES_DATASET = {
    "confirmation_bias": {
        "id": "bias_001",
        "name": "Confirmation Bias",
        "definition": "Tendency to search for, interpret, favor, and recall information that confirms one's preexisting beliefs or hypotheses",
        "category": "belief_formation",
        "severity": "high",
        "impact_score": 0.9,
        "examples": {
            "professional": [
                "Only hiring candidates who confirm existing team culture",
                "Selecting research data that supports predetermined conclusions",
                "Dismissal of client feedback that contradicts project assumptions"
            ],
            "personal": [
                "Only reading news sources that align with political views",
                "Ignoring medical symptoms that don't match self-diagnosis",
                "Continuing to follow investment advice that has been consistently wrong"
            ],
            "academic": [
                "Citing only studies that support hypothesis",
                "Interpreting ambiguous results in favor of expected outcome",
                "Rejecting peer review feedback that challenges methodology"
            ]
        },
        "related_concepts": [
            "selective_perception",
            "biased_interpretation", 
            "belief_persistence",
            "motivated_reasoning",
            "desired_confirmation"
        ],
        "contexts": {
            "decision_making": {
                "frequency": 0.85,
                "severity": "high",
                "intervention_strategies": ["seek_contrarian_views", "use_structured_decision_frameworks"]
            },
            "information_processing": {
                "frequency": 0.75,
                "severity": "medium",
                "intervention_strategies": ["fact_checking", "diverse_information_sources"]
            },
            "belief_systems": {
                "frequency": 0.70,
                "severity": "high",
                "intervention_strategies": ["cognitive_reframing", "evidence_evaluation"]
            }
        },
        "detection_patterns": [
            "dismissing contradictory evidence without proper evaluation",
            "actively seeking information that supports existing beliefs",
            "memory bias favoring confirming information",
            "interpretational bias in ambiguous situations"
        ],
        "mitigation_techniques": [
            "devil's advocate approach",
            "structured information gathering protocols",
            "bias training and awareness programs",
            "peer review and diverse perspectives"
        ],
        "test_queries": [
            "How can confirmation bias be detected in data analysis workflows?",
            "What are the best practices to avoid confirmation bias in research?",
            "How does confirmation bias affect machine learning model selection?"
        ]
    },
    
    "availability_heuristic": {
        "id": "bias_002", 
        "name": "Availability Heuristic",
        "definition": "Overestimating likelihood of events based on how easily examples come to mind",
        "category": "judgment_heuristic",
        "severity": "medium",
        "impact_score": 0.7,
        "examples": {
            "professional": [
                "Overestimating likelihood of project risks based on recent failures",
                "Basing marketing strategy on most memorable customer complaints",
                "Assessing employee performance based on recent memorable incidents"
            ],
            "personal": [
                "Fearing plane crashes more than car accidents due to news coverage",
                "Overestimating likelihood of shark attacks after watching documentaries",
                "Believing job market is poor after hearing about recent layoffs"
            ],
            "academic": [
                "Overestimating probability of rare diseases after case study review",
                "Basing risk assessment on most memorable historical examples",
                "Evaluating student performance based on most recent assignments"
            ]
        },
        "related_concepts": [
            "recency_bias",
            "frequency_estimation",
            "emotional_response",
            "salience_effect",
            "memory_accessibility"
        ],
        "contexts": {
            "risk_assessment": {
                "frequency": 0.80,
                "severity": "medium", 
                "intervention_strategies": ["statistical_analysis", "base_rate_information"]
            },
            "probability_judgment": {
                "frequency": 0.75,
                "severity": "medium",
                "intervention_strategies": ["data_driven_estimation", "expert_probability"]
            },
            "memory_recall": {
                "frequency": 0.70,
                "severity": "low",
                "intervention_strategies": ["structured_recall", "systematic_information_gathering"]
            }
        },
        "detection_patterns": [
            "overestimating frequency of memorable events",
            "underestimating mundane but common occurrences", 
            "basing judgments on ease of recall rather than actual probability",
            "emotional events given disproportionate weight in estimates"
        ],
        "mitigation_techniques": [
            "statistical base rates consideration",
            "systematic information gathering",
            "emotional regulation techniques",
            "structured risk assessment frameworks"
        ],
        "test_queries": [
            "How does availability heuristic affect risk perception in business decisions?",
            "What techniques can reduce availability bias in forecasting?",
            "How to account for availability bias in market research?"
        ]
    },
    
    "anchoring_bias": {
        "id": "bias_003",
        "name": "Anchoring Bias", 
        "definition": "Over-relying on first piece of information received (the 'anchor') when making decisions",
        "category": "reference_point",
        "severity": "medium",
        "impact_score": 0.6,
        "examples": {
            "professional": [
                "Negotiations heavily influenced by initial price offers",
                "Performance evaluations anchored to first impression",
                "Project timelines set based on initial rough estimates"
            ],
            "personal": [
                "House prices judged relative to first house viewed",
                "Salary expectations based on first job offer received",
                "Risk assessments anchored to worst-case scenario initially considered"
            ],
            "academic": [
                "Grading papers influenced by first student's performance",
                "Research conclusions anchored to preliminary findings",
                "Peer review judgments influenced by first reviewer comment"
            ]
        },
        "related_concepts": [
            "first_impression",
            "reference_point",
            "adjustment_bias",
            "primacy_effect",
            "initial_anchor"
        ],
        "contexts": {
            "negotiation": {
                "frequency": 0.85,
                "severity": "high",
                "intervention_strategies": ["multiple_anchors", "structured_anchoring"]
            },
            "estimation": {
                "frequency": 0.70,
                "severity": "medium",
                "intervention_strategies": ["independent_estimates", "range_estimation"]
            },
            "judgment": {
                "frequency": 0.65,
                "severity": "medium",
                "intervention_strategies": ["delayed_judgment", "multiple_perspectives"]
            }
        },
        "detection_patterns": [
            "insufficient adjustment from initial reference point",
            "final decisions clustering around initial values",
            "resistance to updating judgments based on new information",
            "influence of arbitrary anchors on final estimates"
        ],
        "mitigation_techniques": [
            "multiple independent estimates",
            "delayed final judgments",
            "explicit consideration of alternative anchors",
            "structured estimation processes"
        ],
        "test_queries": [
            "How does anchoring bias affect pricing strategies in negotiations?",
            "What methods can reduce anchoring bias in forecasting?",
            "How to identify and mitigate anchoring bias in performance reviews?"
        ]
    },
    
    "overconfidence_bias": {
        "id": "bias_004",
        "name": "Overconfidence Bias",
        "definition": "Excessive confidence in one's own answers, abilities, or judgments",
        "category": "self_assessment",
        "severity": "high",
        "impact_score": 0.85,
        "examples": {
            "professional": [
                "Overestimating accuracy of predictions and forecasts",
                "Underestimating time needed to complete complex projects", 
                "Taking on initiatives beyond actual capability without support"
            ],
            "personal": [
                "Overestimating driving ability compared to others",
                "Believing in ability to predict future events",
                "Overestimating knowledge in unfamiliar domains"
            ],
            "academic": [
                "Overestimating accuracy of research conclusions",
                "Underestimating time for literature review completion",
                "Overestimating capability to master new methodologies"
            ]
        },
        "related_concepts": [
            "dunning_kruger_effect",
            "self_attribution_bias",
            "planning_fallacy",
            "illusion_of_control",
            "superiority_bias"
        ],
        "contexts": {
            "skill_assessment": {
                "frequency": 0.80,
                "severity": "high",
                "intervention_strategies": ["objective_feedback", "skill_testing"]
            },
            "planning": {
                "frequency": 0.75,
                "severity": "high", 
                "intervention_strategies": ["buffer_time", "scenario_planning"]
            },
            "performance_evaluation": {
                "frequency": 0.70,
                "severity": "medium",
                "intervention_strategies": ["360_feedback", "objective_metrics"]
            }
        },
        "detection_patterns": [
            "narrow confidence intervals around estimates",
            "overestimation of relative performance",
            "underestimation of task difficulty",
            "excessive certainty in uncertain situations"
        ],
        "mitigation_techniques": [
            "calibration training",
            "objective feedback mechanisms",
            "uncertainty quantification",
            "peer comparison and benchmarking"
        ],
        "test_queries": [
            "How does overconfidence bias affect project planning and execution?",
            "What are effective ways to measure and reduce overconfidence?",
            "How to identify overconfidence in AI/ML model predictions?"
        ]
    },
    
    "sunk_cost_fallacy": {
        "id": "bias_005",
        "name": "Sunk Cost Fallacy",
        "definition": "Continuing a behavior or endeavor due to previously invested resources (time, money, effort)",
        "category": "investment_effect",
        "severity": "high",
        "impact_score": 0.8,
        "examples": {
            "professional": [
                "Continuing failing projects due to significant past investment",
                "Sticking with outdated technology because of existing infrastructure",
                "Maintaining underperforming employees due to training investment"
            ],
            "personal": [
                "Continuing to watch bad movies to justify ticket cost",
                "Maintaining bad relationships due to time invested",
                "Continuing to play losing poker to 'win back' losses"
            ],
            "academic": [
                "Pursuing research directions despite negative preliminary results",
                "Continuing with dissertation chapters despite evidence of failure",
                "Sticking with peer review process despite multiple rejections"
            ]
        },
        "related_concepts": [
            "loss_aversion",
            "escalation_of_commitment",
            "investment_bias", 
            "commitment_escalation",
            "non_recoverable_costs"
        ],
        "contexts": {
            "economic_decisions": {
                "frequency": 0.75,
                "severity": "high",
                "intervention_strategies": ["cost_benefit_analysis", "exit_criteria"]
            },
            "relationship_management": {
                "frequency": 0.70,
                "severity": "high",
                "intervention_strategies": ["value_assessment", "future_focused_planning"]
            },
            "project_management": {
                "frequency": 0.65,
                "severity": "medium",
                "intervention_strategies": ["milestone_reviews", "objective_assessment"]
            }
        },
        "detection_patterns": [
            "continuing failing endeavors due to past investment",
            "confusing sunk costs with future benefits",
            "resistance to abandoning projects despite poor prospects",
            "escalating commitment to avoid admitting past mistakes"
        ],
        "mitigation_techniques": [
            "focus on future costs and benefits",
            "establishment of clear exit criteria",
            "regular reassessment of project viability",
            "separation of decision-makers from past investment"
        ],
        "test_queries": [
            "How does sunk cost fallacy affect business investment decisions?",
            "What strategies can help overcome sunk cost bias in project management?",
            "How to identify and address sunk cost fallacy in career decisions?"
        ]
    },
    
    "groupthink": {
        "id": "bias_006",
        "name": "Groupthink",
        "definition": "Psychological phenomenon where desire for group consensus overrides realistic appraisal of alternatives",
        "category": "group_dynamics",
        "severity": "high",
        "impact_score": 0.9,
        "examples": {
            "professional": [
                "Teams avoiding conflict by not challenging poor decisions",
                "Conformity to group opinion without critical evaluation",
                "Suppression of dissenting viewpoints in meetings"
            ],
            "personal": [
                "Following crowd behavior without individual assessment",
                "Adopting group opinions without personal research",
                "Avoiding expressing unpopular but correct viewpoints"
            ],
            "academic": [
                "Peer review groups reaching premature consensus",
                "Research teams not challenging established methodologies",
                "Academic committees conforming to dominant paradigm"
            ]
        },
        "related_concepts": [
            "conformity_bias",
            "social_pressure",
            "consensus_seeking",
            "divergent_thinking_suppression",
            "critical_thinking_inhibition"
        ],
        "contexts": {
            "team_decision_making": {
                "frequency": 0.80,
                "severity": "high",
                "intervention_strategies": ["devil_advocate", "anonymous_feedback"]
            },
            "group_processes": {
                "frequency": 0.75,
                "severity": "high",
                "intervention_strategies": ["structured_discussion", "diversity_promotion"]
            },
            "consensus_building": {
                "frequency": 0.70,
                "severity": "medium",
                "intervention_strategies": ["independent_evaluation", "conflict_encouragement"]
            }
        },
        "detection_patterns": [
            "pressure to conform overriding individual judgment",
            "self-censorship of dissenting opinions",
            "illusion of unanimity in group decisions",
            "mindguards protecting group from dissenting information"
        ],
        "mitigation_techniques": [
            "devil's advocate assignments",
            "anonymous feedback mechanisms",
            "diverse group composition",
            "structured decision-making processes"
        ],
        "test_queries": [
            "How does groupthink affect organizational decision-making quality?",
            "What techniques can prevent groupthink in team settings?",
            "How to encourage critical thinking in group environments?"
        ]
    }
}

# Тестовые сценарии для комплексного тестирования
COMPREHENSIVE_BIAS_SCENARIOS = {
    "scenario_1_data_science_team": {
        "name": "Data Science Team Bias Assessment",
        "description": "Testing detection of multiple biases in a data science project",
        "team_context": {
            "size": 5,
            "roles": ["data_scientist", "ml_engineer", "business_analyst", "project_manager", "domain_expert"],
            "project": "Customer churn prediction model"
        },
        "biases_present": ["confirmation_bias", "anchoring_bias", "overconfidence_bias"],
        "expected_detections": [
            "Team only validates model with positive results",
            "Initial accuracy estimate becomes unwavering anchor",
            "Team overestimates model's real-world performance"
        ],
        "test_steps": [
            "Load team data and project context",
            "Simulate bias-prone decisions",
            "Run bias detection algorithms", 
            "Evaluate detection accuracy",
            "Generate bias mitigation recommendations"
        ]
    },
    
    "scenario_2_startup_planning": {
        "name": "Startup Planning Bias Analysis",
        "description": "Evaluating bias impact on startup strategic decisions",
        "startup_context": {
            "stage": "seed",
            "industry": "fintech",
            "funding_stage": "Series A preparation",
            "team_size": 12
        },
        "biases_present": ["sunk_cost_fallacy", "overconfidence_bias", "groupthink"],
        "expected_detections": [
            "Continuing with failing product feature due to investment",
            "Overestimating market size and growth potential", 
            "Group consensus on strategy without critical analysis"
        ],
        "test_steps": [
            "Input startup historical decisions and context",
            "Analyze decision patterns for bias indicators",
            "Compare decisions with optimal rational choices",
            "Calculate bias impact on business outcomes",
            "Recommend bias-aware decision frameworks"
        ]
    },
    
    "scenario_3_research_methodology": {
        "name": "Research Methodology Bias Review",
        "description": "Assessing bias in academic research design and interpretation",
        "research_context": {
            "field": "psychology",
            "study_type": "experimental",
            "sample_size": 150,
            "duration": "6 months"
        },
        "biases_present": ["confirmation_bias", "availability_heuristic", "groupthink"],
        "expected_detections": [
            "Interpretation of ambiguous results toward hypothesis",
            "Overestimation of rare phenomena frequency",
            "Research team consensus without adequate peer review"
        ],
        "test_steps": [
            "Analyze research proposal for bias risks",
            "Review methodology for bias-prone decisions",
            "Simulate results interpretation with different bias levels",
            "Evaluate peer review process for bias indicators",
            "Recommend bias-resistant research protocols"
        ]
    }
}

# Валидационные наборы для тестирования качества знаний
KNOWLEDGE_QUALITY_DATASETS = {
    "high_quality_concepts": {
        "definition": "Well-defined, unambiguous, evidence-based concepts",
        "examples": [
            {
                "name": "artificial_neural_network",
                "definition": "Computing system inspired by biological neural networks",
                "properties": {
                    "clarity": 0.95,
                    "evidence_level": "peer_reviewed",
                    "consensus": 0.90,
                    "testability": 0.88,
                    "falsifiability": 0.85
                },
                "relationships": [
                    {"relation": "is_subset_of", "target": "artificial_intelligence", "confidence": 0.95},
                    {"relation": "uses", "target": "machine_learning", "confidence": 0.80}
                ]
            }
        ],
        "quality_metrics": {
            "semantic_coherence": 0.92,
            "empirical_support": 0.88,
            "conceptual_precision": 0.90,
            "temporal_stability": 0.85
        }
    },
    
    "medium_quality_concepts": {
        "definition": "Partially defined concepts with some evidence base",
        "examples": [
            {
                "name": "emergent_intelligence",
                "definition": "Intelligence that emerges from complex system interactions",
                "properties": {
                    "clarity": 0.70,
                    "evidence_level": "preliminary",
                    "consensus": 0.65,
                    "testability": 0.60,
                    "falsifiability": 0.55
                },
                "relationships": [
                    {"relation": "related_to", "target": "artificial_intelligence", "confidence": 0.70}
                ]
            }
        ],
        "quality_metrics": {
            "semantic_coherence": 0.68,
            "empirical_support": 0.55,
            "conceptual_precision": 0.62,
            "temporal_stability": 0.50
        }
    },
    
    "low_quality_concepts": {
        "definition": "Poorly defined or speculative concepts",
        "examples": [
            {
                "name": "digital_consciousness",
                "definition": "Consciousness in digital systems",
                "properties": {
                    "clarity": 0.30,
                    "evidence_level": "theoretical",
                    "consensus": 0.25,
                    "testability": 0.20,
                    "falsifiability": 0.15
                },
                "relationships": [
                    {"relation": "potentially_related_to", "target": "artificial_intelligence", "confidence": 0.30}
                ]
            }
        ],
        "quality_metrics": {
            "semantic_coherence": 0.25,
            "empirical_support": 0.15,
            "conceptual_precision": 0.20,
            "temporal_stability": 0.10
        }
    }
}