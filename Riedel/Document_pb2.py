# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Document.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Document.proto',
  package='Riedel',
  syntax='proto2',
  serialized_options=_b('\n$cc.refectorie.proj.relation.protobufB\016DocumentProtos'),
  serialized_pb=_b('\n\x0e\x44ocument.proto\x12\x06Riedel\"\xdf\x03\n\x08\x44ocument\x12\x10\n\x08\x66ilename\x18\x01 \x02(\t\x12,\n\tsentences\x18\x02 \x03(\x0b\x32\x19.Riedel.Document.Sentence\x1a\x89\x01\n\x08Sentence\x12&\n\x06tokens\x18\x01 \x03(\x0b\x32\x16.Riedel.Document.Token\x12*\n\x08mentions\x18\x02 \x03(\x0b\x32\x18.Riedel.Document.Mention\x12)\n\x07\x64\x65pTree\x18\x03 \x01(\x0b\x32\x18.Riedel.Document.DepTree\x1a/\n\x05Token\x12\x0c\n\x04word\x18\x01 \x02(\t\x12\x0b\n\x03tag\x18\x02 \x01(\t\x12\x0b\n\x03ner\x18\x03 \x01(\t\x1aR\n\x07Mention\x12\n\n\x02id\x18\x01 \x02(\x05\x12\x12\n\nentityGuid\x18\x02 \x01(\t\x12\x0c\n\x04\x66rom\x18\x03 \x02(\x05\x12\n\n\x02to\x18\x04 \x02(\x05\x12\r\n\x05label\x18\x05 \x02(\t\x1a\x36\n\x07\x44\x65pTree\x12\x0c\n\x04root\x18\x01 \x02(\x05\x12\x0c\n\x04head\x18\x02 \x03(\x05\x12\x0f\n\x07relType\x18\x03 \x03(\t\x1aJ\n\x0fRelationMention\x12\n\n\x02id\x18\x01 \x02(\x05\x12\x0e\n\x06source\x18\x02 \x02(\x05\x12\x0c\n\x04\x64\x65st\x18\x03 \x02(\x05\x12\r\n\x05label\x18\x04 \x02(\t\"\xe4\x01\n\x08Relation\x12\x12\n\nsourceGuid\x18\x01 \x02(\t\x12\x10\n\x08\x64\x65stGuid\x18\x02 \x02(\t\x12\x0f\n\x07relType\x18\x03 \x02(\t\x12\x34\n\x07mention\x18\x04 \x03(\x0b\x32#.Riedel.Relation.RelationMentionRef\x1ak\n\x12RelationMentionRef\x12\x10\n\x08\x66ilename\x18\x01 \x02(\t\x12\x10\n\x08sourceId\x18\x02 \x02(\x05\x12\x0e\n\x06\x64\x65stId\x18\x03 \x02(\x05\x12\x0f\n\x07\x66\x65\x61ture\x18\x04 \x03(\t\x12\x10\n\x08sentence\x18\x05 \x01(\t\"\xb5\x01\n\x06\x45ntity\x12\x0c\n\x04guid\x18\x01 \x02(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x0c\n\x04pred\x18\x04 \x01(\t\x12\x30\n\x07mention\x18\x05 \x03(\x0b\x32\x1f.Riedel.Entity.EntityMentionRef\x1a\x41\n\x10\x45ntityMentionRef\x12\x10\n\x08\x66ilename\x18\x01 \x02(\t\x12\n\n\x02id\x18\x02 \x02(\x05\x12\x0f\n\x07\x66\x65\x61ture\x18\x03 \x03(\tB6\n$cc.refectorie.proj.relation.protobufB\x0e\x44ocumentProtos')
)




_DOCUMENT_SENTENCE = _descriptor.Descriptor(
  name='Sentence',
  full_name='Riedel.Document.Sentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tokens', full_name='Riedel.Document.Sentence.tokens', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mentions', full_name='Riedel.Document.Sentence.mentions', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='depTree', full_name='Riedel.Document.Sentence.depTree', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=104,
  serialized_end=241,
)

_DOCUMENT_TOKEN = _descriptor.Descriptor(
  name='Token',
  full_name='Riedel.Document.Token',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='word', full_name='Riedel.Document.Token.word', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tag', full_name='Riedel.Document.Token.tag', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ner', full_name='Riedel.Document.Token.ner', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=243,
  serialized_end=290,
)

_DOCUMENT_MENTION = _descriptor.Descriptor(
  name='Mention',
  full_name='Riedel.Document.Mention',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Riedel.Document.Mention.id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='entityGuid', full_name='Riedel.Document.Mention.entityGuid', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='from', full_name='Riedel.Document.Mention.from', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='to', full_name='Riedel.Document.Mention.to', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='Riedel.Document.Mention.label', index=4,
      number=5, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=292,
  serialized_end=374,
)

_DOCUMENT_DEPTREE = _descriptor.Descriptor(
  name='DepTree',
  full_name='Riedel.Document.DepTree',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='root', full_name='Riedel.Document.DepTree.root', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='head', full_name='Riedel.Document.DepTree.head', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relType', full_name='Riedel.Document.DepTree.relType', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=376,
  serialized_end=430,
)

_DOCUMENT_RELATIONMENTION = _descriptor.Descriptor(
  name='RelationMention',
  full_name='Riedel.Document.RelationMention',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Riedel.Document.RelationMention.id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='source', full_name='Riedel.Document.RelationMention.source', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dest', full_name='Riedel.Document.RelationMention.dest', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='Riedel.Document.RelationMention.label', index=3,
      number=4, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=432,
  serialized_end=506,
)

_DOCUMENT = _descriptor.Descriptor(
  name='Document',
  full_name='Riedel.Document',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='Riedel.Document.filename', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentences', full_name='Riedel.Document.sentences', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DOCUMENT_SENTENCE, _DOCUMENT_TOKEN, _DOCUMENT_MENTION, _DOCUMENT_DEPTREE, _DOCUMENT_RELATIONMENTION, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=506,
)


_RELATION_RELATIONMENTIONREF = _descriptor.Descriptor(
  name='RelationMentionRef',
  full_name='Riedel.Relation.RelationMentionRef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='Riedel.Relation.RelationMentionRef.filename', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sourceId', full_name='Riedel.Relation.RelationMentionRef.sourceId', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='destId', full_name='Riedel.Relation.RelationMentionRef.destId', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature', full_name='Riedel.Relation.RelationMentionRef.feature', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentence', full_name='Riedel.Relation.RelationMentionRef.sentence', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=630,
  serialized_end=737,
)

_RELATION = _descriptor.Descriptor(
  name='Relation',
  full_name='Riedel.Relation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sourceGuid', full_name='Riedel.Relation.sourceGuid', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='destGuid', full_name='Riedel.Relation.destGuid', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relType', full_name='Riedel.Relation.relType', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mention', full_name='Riedel.Relation.mention', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_RELATION_RELATIONMENTIONREF, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=509,
  serialized_end=737,
)


_ENTITY_ENTITYMENTIONREF = _descriptor.Descriptor(
  name='EntityMentionRef',
  full_name='Riedel.Entity.EntityMentionRef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='Riedel.Entity.EntityMentionRef.filename', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='Riedel.Entity.EntityMentionRef.id', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature', full_name='Riedel.Entity.EntityMentionRef.feature', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=856,
  serialized_end=921,
)

_ENTITY = _descriptor.Descriptor(
  name='Entity',
  full_name='Riedel.Entity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='guid', full_name='Riedel.Entity.guid', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='Riedel.Entity.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='Riedel.Entity.type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pred', full_name='Riedel.Entity.pred', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mention', full_name='Riedel.Entity.mention', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ENTITY_ENTITYMENTIONREF, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=740,
  serialized_end=921,
)

_DOCUMENT_SENTENCE.fields_by_name['tokens'].message_type = _DOCUMENT_TOKEN
_DOCUMENT_SENTENCE.fields_by_name['mentions'].message_type = _DOCUMENT_MENTION
_DOCUMENT_SENTENCE.fields_by_name['depTree'].message_type = _DOCUMENT_DEPTREE
_DOCUMENT_SENTENCE.containing_type = _DOCUMENT
_DOCUMENT_TOKEN.containing_type = _DOCUMENT
_DOCUMENT_MENTION.containing_type = _DOCUMENT
_DOCUMENT_DEPTREE.containing_type = _DOCUMENT
_DOCUMENT_RELATIONMENTION.containing_type = _DOCUMENT
_DOCUMENT.fields_by_name['sentences'].message_type = _DOCUMENT_SENTENCE
_RELATION_RELATIONMENTIONREF.containing_type = _RELATION
_RELATION.fields_by_name['mention'].message_type = _RELATION_RELATIONMENTIONREF
_ENTITY_ENTITYMENTIONREF.containing_type = _ENTITY
_ENTITY.fields_by_name['mention'].message_type = _ENTITY_ENTITYMENTIONREF
DESCRIPTOR.message_types_by_name['Document'] = _DOCUMENT
DESCRIPTOR.message_types_by_name['Relation'] = _RELATION
DESCRIPTOR.message_types_by_name['Entity'] = _ENTITY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Document = _reflection.GeneratedProtocolMessageType('Document', (_message.Message,), dict(

  Sentence = _reflection.GeneratedProtocolMessageType('Sentence', (_message.Message,), dict(
    DESCRIPTOR = _DOCUMENT_SENTENCE,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Document.Sentence)
    ))
  ,

  Token = _reflection.GeneratedProtocolMessageType('Token', (_message.Message,), dict(
    DESCRIPTOR = _DOCUMENT_TOKEN,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Document.Token)
    ))
  ,

  Mention = _reflection.GeneratedProtocolMessageType('Mention', (_message.Message,), dict(
    DESCRIPTOR = _DOCUMENT_MENTION,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Document.Mention)
    ))
  ,

  DepTree = _reflection.GeneratedProtocolMessageType('DepTree', (_message.Message,), dict(
    DESCRIPTOR = _DOCUMENT_DEPTREE,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Document.DepTree)
    ))
  ,

  RelationMention = _reflection.GeneratedProtocolMessageType('RelationMention', (_message.Message,), dict(
    DESCRIPTOR = _DOCUMENT_RELATIONMENTION,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Document.RelationMention)
    ))
  ,
  DESCRIPTOR = _DOCUMENT,
  __module__ = 'Document_pb2'
  # @@protoc_insertion_point(class_scope:Riedel.Document)
  ))
_sym_db.RegisterMessage(Document)
_sym_db.RegisterMessage(Document.Sentence)
_sym_db.RegisterMessage(Document.Token)
_sym_db.RegisterMessage(Document.Mention)
_sym_db.RegisterMessage(Document.DepTree)
_sym_db.RegisterMessage(Document.RelationMention)

Relation = _reflection.GeneratedProtocolMessageType('Relation', (_message.Message,), dict(

  RelationMentionRef = _reflection.GeneratedProtocolMessageType('RelationMentionRef', (_message.Message,), dict(
    DESCRIPTOR = _RELATION_RELATIONMENTIONREF,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Relation.RelationMentionRef)
    ))
  ,
  DESCRIPTOR = _RELATION,
  __module__ = 'Document_pb2'
  # @@protoc_insertion_point(class_scope:Riedel.Relation)
  ))
_sym_db.RegisterMessage(Relation)
_sym_db.RegisterMessage(Relation.RelationMentionRef)

Entity = _reflection.GeneratedProtocolMessageType('Entity', (_message.Message,), dict(

  EntityMentionRef = _reflection.GeneratedProtocolMessageType('EntityMentionRef', (_message.Message,), dict(
    DESCRIPTOR = _ENTITY_ENTITYMENTIONREF,
    __module__ = 'Document_pb2'
    # @@protoc_insertion_point(class_scope:Riedel.Entity.EntityMentionRef)
    ))
  ,
  DESCRIPTOR = _ENTITY,
  __module__ = 'Document_pb2'
  # @@protoc_insertion_point(class_scope:Riedel.Entity)
  ))
_sym_db.RegisterMessage(Entity)
_sym_db.RegisterMessage(Entity.EntityMentionRef)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)