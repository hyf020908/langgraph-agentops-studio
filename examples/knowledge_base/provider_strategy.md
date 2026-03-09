# Provider Strategy Notes

Production agent systems should keep provider settings configuration-driven. A practical
approach is to use OpenAI-compatible clients with pluggable base URLs so teams can switch
between OpenAI, DeepSeek, or gateway-managed endpoints with minimal code changes.
