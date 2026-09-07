import json

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from support_troubleshooting_agent.agents import recommendation as agent


def plan():
    return {
        'recovery_test': 'Verify a valid login succeeds and a token is issued.',
        'uncertainty': 'The reported certificate rotation needs confirmation.',
        'immediate_mitigation': [{
            'evidence_source': 'ticket',
            'action': 'Confirm the trusted certificate matches the rotated identity-provider certificate.',
            'rationale': 'Sign-in failures followed certificate rotation.',
            'verification': 'Compare the currently trusted certificate with the provider certificate.',
            'recovery': 'Read-only inspection requires no rollback.',
        }],
        'root_cause_remediation': [{
            'evidence_source': 'retrieved_evidence',
            'action': 'If mismatched, update the trusted certificate and reload authentication.',
            'rationale': 'A mismatched trust configuration rejects otherwise valid logins.',
            'verification': 'Verify a valid login succeeds and a token is issued.',
            'recovery': 'If reload worsens failures, restore the last valid trust configuration.',
        }],
        'preventive_actions': [{'action': 'Validate trust configuration during future certificate rotations.',
                                'rationale': 'This detects mismatches before users lose access.'}],
        'escalation_guidance': ['Escalate unresolved trust mismatches to the identity-service owner.'],
    }


def test_context_and_paired_actions_preserve_existing_contract(monkeypatch):
    captured = []

    class Model:
        def with_structured_output(self, schema, method):
            assert method == 'json_schema'
            def invoke(prompt):
                captured.append(prompt.to_messages())
                value = ({'remedy_type': 'refresh_rotated_trust', 'component': 'identity provider',
                          'evidence_source': 'ticket'} if schema is not agent.RecoveryTest else
                         {'recovery_test': 'Verify a valid login succeeds and a token is issued.'})
                return schema.model_validate(value)
            return RunnableLambda(invoke)

    monkeypatch.setattr(agent, 'get_chat_model', lambda: Model())
    state = {
        'ticket': 'Login fails after the identity provider certificate rotation.',
        'ticket_summary': 'Authentication failure',
        'root_cause': {'primary_cause': 'Trust mismatch', 'confidence': 0.6},
        'retrieved_documents': [{'content': 'Reload trust after updating the certificate.', 'source': 'auth-runbook'}],
        'log_analysis': {'summary': 'Trust validation failed'},
        'investigation_decision': 'use_retrieved_knowledge',
        'errors': [{'step': 'log_agent', 'user_message': 'One log source missing'}],
        'expected_recommendation': 'MUST NOT REACH PROMPT',
        'api_key': 'MUST NOT REACH PROMPT',
    }
    original = json.dumps(state, sort_keys=True)
    result = agent.recommendation_agent(state)
    messages = captured[0]
    context = json.loads(messages[1].content.split('\n', 1)[1])
    assert context['ticket'] == state['ticket']
    assert 'diagnosis' not in context
    assert json.dumps(state['root_cause']) in captured[1][1].content
    assert context['retrieved_evidence'] == state['retrieved_documents']
    assert context['log_findings'] == state['log_analysis']
    assert context['investigation']['investigation_decision'] == 'use_retrieved_knowledge'
    assert 'provisional' in context['confidence_policy']
    assert 'MUST NOT REACH PROMPT' not in messages[1].content
    assert json.dumps(state, sort_keys=True) == original
    rec = result['recommendations']
    assert set(rec) == {'recommended_actions', 'validation_steps', 'escalation_guidance', 'preventative_measures'}
    assert len(rec['validation_steps']) == 2
    for identifier in ('M1', 'R1'):
        action = next(text for text in rec['recommended_actions'] if f'[{identifier}]' in text)
        assert 'Why:' in action and 'Recovery:' in action
        assert any(f'[{identifier}]' in text for text in rec['validation_steps'])
    assert 'login succeeds' in rec['validation_steps'][1]
    assert 'root_cause' not in result


@pytest.mark.parametrize('confidence', [None, 0.3, 0.79, float('nan'), True, 'invalid', 5])
def test_unreliable_confidence_is_provisional(confidence):
    context = agent._context({'root_cause': {'confidence': confidence}})
    assert 'provisional' in context['confidence_policy']


@pytest.mark.parametrize('missing', ['action', 'rationale', 'verification', 'recovery'])
def test_remediation_cannot_omit_required_fields(missing):
    value = plan()
    del value['root_cause_remediation'][0][missing]
    with pytest.raises(ValidationError):
        agent.RecommendationPlan.model_validate(value)


def test_invalid_or_unavailable_plan_does_not_invent_checkout_fix(monkeypatch):
    def unavailable():
        raise RuntimeError('Model unavailable')
    monkeypatch.setattr(agent, 'get_chat_model', unavailable)
    result = agent.recommendation_agent({'ticket': 'Notifications fail because the TLS certificate expired.', 'root_cause': {'confidence': 0.4}})
    text = json.dumps(result['recommendations']).lower()
    assert 'checkout' not in text and 'database' not in text and 'deployment' not in text
    assert 'no production change' in text
    assert result['errors'][-1]['step'] == 'recommendation_agent'


def test_missing_ticket_fallback_requests_ticket():
    result = agent.recommendation_agent({})
    assert 'original support ticket' in json.dumps(result['recommendations'])
    assert 'No root-cause data' in result['errors'][-1]['message']


def test_grounding_rejects_absent_source_and_invented_setting():
    value = plan()
    context = agent._context({'ticket': 'Login fails after certificate rotation.'})
    with pytest.raises(ValueError, match='empty'):
        agent._validate_grounding(agent.RecommendationPlan.model_validate(value), context)
    value['root_cause_remediation'][0]['evidence_source'] = 'ticket'
    value['root_cause_remediation'][0]['action'] = 'Increase the timeout to 42 seconds.'
    with pytest.raises(ValueError, match='numeric'):
        agent._validate_grounding(agent.RecommendationPlan.model_validate(value), context)


def test_verification_repairs_invented_target_before_returning(monkeypatch):
    prompts = []
    verification_calls = []

    class Model:
        def with_structured_output(self, schema, method):
            def invoke(prompt):
                prompts.append(prompt.to_messages())
                if schema is agent.RecoveryTest:
                    verification_calls.append(prompt.to_messages())
                    return schema(recovery_test='Verify login succeeds within 42 seconds.' if len(verification_calls) == 1 else 'Verify a valid login succeeds and a token is issued.')
                return schema(remedy_type='refresh_rotated_trust', component='certificate', evidence_source='ticket')
            return RunnableLambda(invoke)

    monkeypatch.setattr(agent, 'get_chat_model', lambda: Model())
    result = agent.recommendation_agent({'ticket': 'Login fails after certificate rotation.',
                                         'root_cause': {'confidence': 0.6}})
    assert len(prompts) == 3
    assert 'validation error' in verification_calls[1][-1].content
    assert '42 seconds' not in json.dumps(result['recommendations'])
    assert 'Provisional diagnosis' in result['recommendations']['recommended_actions'][0]
    assert 'login succeeds' in result['recommendations']['validation_steps'][1]
    assert 'errors' not in result


def test_invalid_operation_uses_positive_template_without_unbounded_retry(monkeypatch):
    calls = []

    class Model:
        def with_structured_output(self, schema, method):
            def invoke(prompt):
                calls.append(prompt)
                if schema is agent.RecoveryTest:
                    return schema(recovery_test='Verify login succeeds within 42 seconds.')
                return schema(remedy_type='refresh_rotated_trust', component='certificate', evidence_source='ticket')
            return RunnableLambda(invoke)

    monkeypatch.setattr(agent, 'get_chat_model', lambda: Model())
    result = agent.recommendation_agent({'ticket': 'Login fails after certificate rotation.',
                                         'root_cause': {'confidence': 0.6}})
    assert len(calls) == 3
    assert '42 seconds' not in json.dumps(result['recommendations'])
    assert 'verify successful completion' in json.dumps(result['recommendations'])
    assert 'Root-cause remediation' in json.dumps(result['recommendations'])
    assert 'errors' not in result


def test_unsubstantiated_diagnosis_or_generic_runbook_cannot_enable_a_remedy(monkeypatch):
    def unexpected_model_call():
        raise AssertionError('No grounded repair was available')
    monkeypatch.setattr(agent, 'get_chat_model', unexpected_model_call)
    result = agent.recommendation_agent({'ticket': 'The report is slow; details are unavailable.',
        'root_cause': {'primary_cause': 'The disk is full', 'confidence': 0.99},
        'retrieved_documents': [{'content': 'When a disk is full, application writes fail.'}]})
    assert 'No production change' in json.dumps(result['recommendations'])
    assert 'errors' not in result


def test_negated_faults_do_not_enable_matching_change():
    assert 'recover_storage_capacity' not in agent._eligible_patterns('The disk is not full; writes fail for an unknown reason.')
    assert 'renew_expired_certificate' not in agent._eligible_patterns('The certificate has not expired; investigate the API error.')
    assert 'restore_processing' not in agent._eligible_patterns('The service feels slow; no further details are available.')


def test_selection_cannot_invent_component_or_use_wrong_source():
    context = agent._context({'ticket': 'The audit volume is full and writes fail.'})
    schema = agent._selection_schema(['recover_storage_capacity'])
    with pytest.raises(ValueError, match='identifiable'):
        agent._validate_selection(schema(remedy_type='recover_storage_capacity', component='new server', evidence_source='ticket'), context)
    with pytest.raises(ValueError, match='source'):
        agent._validate_selection(schema(remedy_type='recover_storage_capacity', component='audit volume', evidence_source='retrieved_evidence'), context)
    agent._validate_selection(schema(remedy_type='recover_storage_capacity', component='audit volume', evidence_source='ticket'), context)


@pytest.mark.parametrize('target', ['2GB', '42 seconds', '10%'])
def test_grounding_rejects_numeric_values_with_units(target):
    value = plan()
    value['root_cause_remediation'][0]['evidence_source'] = 'ticket'
    value['root_cause_remediation'][0]['action'] = f'Increase the configured limit to {target}.'
    with pytest.raises(ValueError, match='numeric'):
        agent._validate_grounding(agent.RecommendationPlan.model_validate(value),
                                  agent._context({'ticket': 'A request failed with status 142.'}))


@pytest.mark.parametrize('unsafe_test', ['Run curl -k against the API.', 'Disable certificate validation and test.',
                                         'Print the environment variables.'])
def test_recovery_test_cannot_add_commands_or_bypass_checks(unsafe_test):
    value = plan()
    value['recovery_test'] = unsafe_test
    with pytest.raises(ValueError, match='ordinary language'):
        agent._validate_grounding(agent.RecommendationPlan.model_validate(value),
                                  agent._context({'ticket': 'Login failed.'}))


def test_component_description_resolves_to_actual_incident_target():
    context = agent._context({'ticket': 'Connections to the MongoDB primary time out.'})
    schema = agent._selection_schema(['restore_dependency_health'])
    selection = schema(remedy_type='restore_dependency_health', component='MongoDB primary connection', evidence_source='ticket')
    agent._validate_selection(selection, context)
    assert selection.component == 'MongoDB primary'


def test_recovery_cannot_preserve_original_error_as_success():
    with pytest.raises(ValueError, match='failed state'):
        agent._validate_recovery_goal('Verify Nginx returns 504 for POST /checkout after the fix.')
    agent._validate_recovery_goal('Verify Nginx no longer returns 504 for POST /checkout.')
    agent._validate_recovery_goal('Verify no request returns HTTP 503 after the fix.')
    agent._validate_recovery_goal('Verify the API returns 200 after the fix.')
