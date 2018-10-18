from Riedel import Document_pb2

# Iterates though all people in the AddressBook and prints info about them.
def ListDocument(document):
    print(document.filename)
    for sentence in document.sentences:
        pass

    # for person in address_book.people:
    #   print "Person ID:", person.id
    #   print "  Name:", person.name
    #   if person.HasField('email'):
    #     print "  E-mail address:", person.email
    #
    #   for phone_number in person.phones:
    #     if phone_number.type == addressbook_pb2.Person.MOBILE:
    #       print "  Mobile phone #: ",
    #     elif phone_number.type == addressbook_pb2.Person.HOME:
    #       print "  Home phone #: ",
    #     elif phone_number.type == addressbook_pb2.Person.WORK:
    #       print "  Work phone #: ",
    #     print phone_number.number


document = Document_pb2.Document()

# relation = Document_pb2.Relation()
#
# entity = Document_pb2.Entity()

# with open("./relations/kb_manual/testNewRelations.pb", "rb") as f:
#     relation.ParseFromString(f.read())

# with open("./heldout_relations/testPositive.pb", "rb") as f:
#     document.ParseFromString(f.read())

# Read the existing address book.
with open("./manual-05-06/nyt-2005-2006.backup/1638661.xml.pb", "rb") as f:
    document.ParseFromString(f.read())

ListDocument(document)

